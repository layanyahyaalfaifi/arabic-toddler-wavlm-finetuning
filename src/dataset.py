"""Dataset and balanced-batch sampler for AbjadKids fine-tuning."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler


def _extract_speaker(path: str) -> str:
    """Extract a speaker identifier from an AbjadKids filename.

    AbjadKids files follow the pattern ``Label_speaker_uuid.ext``. The speaker
    identifier is the second underscore-separated token after the stem.
    """
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[1].strip().lower()
    return "unknown"


def index_dataset(data_dir: str, categories, audio_extensions) -> pd.DataFrame:
    """Walk the AbjadKids directory tree and build a dataframe of audio files."""
    rows = []
    for category in categories:
        category_dir = Path(data_dir) / category
        if not category_dir.is_dir():
            continue
        for class_dir in category_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label_name = class_dir.name
            for file in class_dir.rglob("*"):
                if file.suffix.lower() in audio_extensions:
                    rows.append(
                        {
                            "path": str(file),
                            "category": category,
                            "label_name": label_name,
                        }
                    )
    df = pd.DataFrame(rows)
    df["speaker"] = df["path"].apply(_extract_speaker)
    return df


def build_splits(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    excluded_classes=None,
):
    """Build speaker-disjoint train / val / test splits.

    No speaker is allowed to appear in more than one split. The label index is
    rebuilt after exclusions so that ``label`` columns contain a contiguous
    range starting from zero.
    """
    if excluded_classes:
        df = df[~df["label_name"].isin(excluded_classes)].reset_index(drop=True)

    label_names = sorted(df["label_name"].unique())
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}
    df = df.copy()
    df["label"] = df["label_name"].map(label2id)

    speakers = df["speaker"].unique()
    train_speakers, temp_speakers = train_test_split(
        speakers, test_size=(1.0 - train_size), random_state=seed
    )
    relative_test_size = test_size / (val_size + test_size)
    val_speakers, test_speakers = train_test_split(
        temp_speakers, test_size=relative_test_size, random_state=seed
    )

    train_df = df[df["speaker"].isin(train_speakers)].reset_index(drop=True)
    val_df = df[df["speaker"].isin(val_speakers)].reset_index(drop=True)
    test_df = df[df["speaker"].isin(test_speakers)].reset_index(drop=True)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "label2id": label2id,
        "id2label": id2label,
    }


class AbjadAudioDataset(Dataset):
    """Audio dataset that loads, resamples, normalises, crops or pads each clip."""

    def __init__(self, dataframe: pd.DataFrame, sr: int = 16000, max_len: int = 40000):
        self.df = dataframe.reset_index(drop=True)
        self.sr = sr
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, path: str) -> np.ndarray:
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        wav = wav.astype(np.float32)
        peak = np.max(np.abs(wav)) + 1e-8
        wav = wav / peak
        if len(wav) > self.max_len:
            start = random.randint(0, len(wav) - self.max_len)
            wav = wav[start : start + self.max_len]
        else:
            pad = self.max_len - len(wav)
            wav = np.pad(wav, (0, pad))
        return wav

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        wav = self._load_audio(row["path"])
        return {
            "input_values": torch.tensor(wav, dtype=torch.float32),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "label_name": row["label_name"],
        }


class BalancedBatchSampler(Sampler):
    """Yield infinite batches with ``n_classes`` distinct labels and ``n_samples`` per label.

    Required for Supervised Contrastive Learning, which needs at least two
    positive pairs per class within each batch.
    """

    def __init__(self, dataframe: pd.DataFrame, n_classes: int = 8, n_samples: int = 4):
        self.df = dataframe.reset_index(drop=True)
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.df["label"]):
            self.label_to_indices[int(label)].append(idx)
        self.labels = list(self.label_to_indices.keys())

    def __iter__(self):
        while True:
            selected_labels = random.sample(self.labels, self.n_classes)
            batch = []
            for label in selected_labels:
                indices = self.label_to_indices[label]
                chosen = random.choices(indices, k=self.n_samples)
                batch.extend(chosen)
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return len(self.df) // self.batch_size


def collate_fn(batch):
    """Stack input tensors and labels for the DataLoader."""
    input_values = torch.stack([x["input_values"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])
    return {"input_values": input_values, "labels": labels}

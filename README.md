# Arabic Toddler WavLM Fine-tuning

Supervised Contrastive Learning fine-tuning of WavLM-base-plus on the AbjadKids
Arabic-children speech corpus, conducted as an ablation experiment for the
HAYATY developmental-screening project.

This repository documents an embedding-backbone selection experiment carried out
during the development of the HAYATY Speech Agent. The fine-tuned model achieved
strong in-domain performance on AbjadKids but exhibited representational
collapse on real spontaneous toddler speech, motivating the final decision to
retain the pretrained WavLM-base-plus (layer 9) as the production backbone.

## Motivation

The HAYATY Speech Agent represents toddler vocalisations as fixed-dimensional
embeddings used for clustering. We hypothesised that supervised contrastive
fine-tuning on AbjadKids (a closed-vocabulary Arabic children corpus) would
tighten the embedding space for same-word vocalisations relative to
the pretrained baseline. The experiment tested this hypothesis directly.

## Dataset

- **AbjadKids** (Hugging Face: `Aziz-snoubra/Abjad-Kids`)
- 40,000+ Arabic-children audio samples
- ~140 word classes (alphabet, colours, numbers)
- Audio sampled at 16 kHz, mono
- Train / Val / Test split: 70% / 15% / 15%, **speaker-disjoint**
- The under-represented class `Walad` was excluded.

## Method

- **Backbone**: `microsoft/wavlm-base-plus` (Hugging Face Transformers)
- **Loss**: Supervised Contrastive Loss (Khosla et al., 2020), τ = 0.05
- **Pooling**: Mean over time
- **Projection head**: Linear (768 → 256) → ReLU → Dropout(0.1) → Linear (256 → 128) → L2-normalise
- **Sampler**: Balanced batch sampler (8 classes × 4 samples per batch = 32)
- **Frozen layers**: First 4 transformer layers (low-level acoustic features preserved)
- **Optimiser**: AdamW (lr = 1e-5, weight decay = 1e-4, β = (0.9, 0.98))
- **Schedule**: Cosine annealing over 2,000 steps, η_min = 1e-6
- **Gradient clipping**: max-norm 1.0
- **Hardware**: single Tesla T4 GPU (Google Colab)

## In-Domain Results (AbjadKids test set)

| Metric                           | Value |
| -------------------------------- | -----:|
| Top-1 retrieval accuracy         |  81.4% |
| Top-5 retrieval accuracy         |  89.5% |
| Same-word cosine similarity      |  0.93 |
| Different-word cosine similarity |  0.67 |
| Same/Different separation gap    |  0.26 |
| AHC ARI / NMI / Purity           |  0.33 / 0.76 / 0.49 |
| HDBSCAN ARI / NMI / Purity       |  0.54 / 0.86 / 0.70 |

Speaker-disjoint splits ensure that retrieval is evaluated only on speakers
that the model has never seen during training.

## Out-of-Domain Behaviour

When applied to spontaneous toddler speech recorded in real home conditions,
the fine-tuned embeddings exhibited representational collapse: distinct
vocalisations from the same child were assigned to a single dominant cluster
despite clear acoustic differences. The pretrained baseline produced
separable clusters under the same conditions.

## Diagnosis

The failure mode is consistent with a domain-shift between AbjadKids and
spontaneous toddler speech:

| Property            | AbjadKids                  | Spontaneous toddler speech     |
| ------------------- | -------------------------- | ------------------------------ |
| Vocabulary          | Closed (~140 words)        | Open                           |
| Articulation        | Mature, conscious effort   | Immature, substitutions / omissions |
| Recording           | Clean, studio-style        | Home audio with ambient noise  |
| Speaker age         | ~4-7 years                 | Under 3 years                  |
| Prosody             | Stable (read-speech)       | Highly variable                |

The contrastive objective over-specialised the embedding space to the narrow
distribution of AbjadKids, eroding the broad phonetic generality that
self-supervised pretraining preserves.

## Decision

The fine-tuned model was discarded. The pretrained WavLM-base-plus
(`microsoft/wavlm-base-plus`, layer 9) was retained as the embedding backbone
for the HAYATY Speech Agent.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── .gitignore
├── configs/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── dataset.py        # AbjadAudioDataset and BalancedBatchSampler
│   ├── model.py          # WavLMSupConModel (backbone + projection head)
│   ├── loss.py           # SupConLoss (Khosla et al., 2020)
│   ├── train.py          # Training loop with checkpointing
│   └── evaluate.py       # Retrieval and clustering metrics
├── scripts/
│   ├── prepare_data.py   # Build train/val/test splits
│   └── run_training.py   # End-to-end training entry point
├── notebooks/
│   └── training_notebook.ipynb
└── docs/
    └── results.md
```

## Setup

```bash
git clone https://github.com/layanyahyaalfaifi/arabic-toddler-wavlm-finetuning.git
cd arabic-toddler-wavlm-finetuning
pip install -r requirements.txt
```

The AbjadKids dataset is downloaded directly from Hugging Face the first time
`scripts/prepare_data.py` is executed.

## Usage

Prepare the data splits:

```bash
python scripts/prepare_data.py --output_dir data/
```

Run training:

```bash
python scripts/run_training.py --config configs/default.yaml
```

Evaluate the trained model:

```bash
python -m src.evaluate --checkpoint checkpoints/wavlm_supcon_step_2000.pt
```

## Citation

If you use this code, please cite both this repository and the underlying
references:

```bibtex
@misc{alfaifi2026hayaty,
  title  = {HAYATY: Multi-Agent Developmental Screening for Children},
  author = {Layan Yahya Al-Faifi},
  year   = {2026},
  note   = {Graduation thesis project}
}

@inproceedings{khosla2020supcon,
  title     = {Supervised Contrastive Learning},
  author    = {Khosla, Prannay and Teterwak, Piotr and Wang, Chen and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020}
}

@article{chen2022wavlm,
  title   = {{WavLM}: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing},
  author  = {Chen, Sanyuan and Wang, Chengyi and Chen, Zhuo and others},
  journal = {IEEE Journal of Selected Topics in Signal Processing},
  year    = {2022}
}
```

## Acknowledgements

- Microsoft Research for the WavLM-base-plus model
- The AbjadKids dataset contributors (`Aziz-snoubra/Abjad-Kids` on Hugging Face)
- Khosla et al. for the Supervised Contrastive Learning loss formulation

## License

MIT License — see [LICENSE](LICENSE) for details.

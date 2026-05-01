# Results

## In-Domain Evaluation (AbjadKids Test Set)

The held-out test set contains speakers that the model never saw during
training. All metrics below are reported on this disjoint set.

### Retrieval

| Metric                    | Value |
| ------------------------- | -----:|
| Top-1 retrieval accuracy  |  81.4% |
| Top-5 retrieval accuracy  |  89.5% |

### Embedding Geometry

| Metric                              | Value |
| ----------------------------------- | -----:|
| Mean same-word cosine similarity    |  0.93 |
| Mean different-word cosine similarity |  0.67 |
| Same / Different separation gap     |  0.26 |

### Clustering

| Algorithm                          | ARI  | NMI  | Purity |
| ---------------------------------- | ----:| ----:| ------:|
| Agglomerative (cosine, average)    | 0.33 | 0.76 |   0.49 |
| HDBSCAN (euclidean, min_cluster=5) | 0.54 | 0.86 |   0.70 |

The HDBSCAN figures are reported on the subset of points that received a
non-noise label; HDBSCAN classified roughly half of the points as noise
(label = -1) and these were excluded from the metrics.

## Out-of-Domain Evaluation (Spontaneous Toddler Recordings)

When the same model was applied to spontaneous toddler speech recorded in
real home conditions, the embeddings exhibited representational collapse:
distinct vocalisations from a single child were assigned to a single
dominant cluster despite clear acoustic differences. The pretrained
baseline produced separable clusters under the same conditions.

A formal numeric comparison was not performed in the out-of-domain setting,
because no labelled corpus of spontaneous Arabic toddler speech is
available. The collapse was confirmed through manual auditory inspection of
the resulting clusters.

## Methodological Limitations

This experiment was a comparative ablation, not a production training run.
The following limitations are acknowledged:

- No validation-loss-based early stopping (training ran for a fixed
  schedule of 2,000 steps).
- No baseline ablation against pretrained WavLM with identical evaluation
  pipelines was reported numerically.
- Mean pooling over time was used; attention pooling was not tested.
- No data augmentation was applied.

These limitations did not change the final decision, since the failure on
spontaneous toddler speech reflects a domain mismatch rather than a
training defect: read-speech of a closed vocabulary cannot serve as a
transfer source for spontaneous open-vocabulary toddler vocalisations.

## Decision

The fine-tuned model was discarded. The pretrained WavLM-base-plus
(`microsoft/wavlm-base-plus`, layer 9) was retained as the embedding
backbone for the HAYATY Speech Agent.

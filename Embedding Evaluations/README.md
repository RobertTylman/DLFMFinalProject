# Shazam CLAP Embedding Analysis

This repository evaluates how well two CLAP embedding checkpoints preserve music identity and genre under noisy GTZAN audio augmentations.

## What Is Evaluated

The scripts assume an embedding root with this layout:

```text
Data/
  genres_original/{genre}/{track_id}.npy
  genres_augmented/{noise_type}/{snr}dB/{genre}/{track_id}.npy
```

Ground truth is derived from the dataset paths:

- Genre classification ground truth is the GTZAN genre folder name.
- Exact-song retrieval ground truth is the clean original track with the same `track_id` filename stem as the noisy query.

## Results

The included plots are under `results/`:

- `results/genre_classification/`: train a classifier on clean original embeddings, then test noisy augmented embeddings against the GTZAN folder genre.
- `results/exact_song_retrieval/`: query noisy augmented embeddings against a clean original-track index and score whether the retrieved clean track has the same `track_id`.
- `results/data_overview/`: dataset count checks by genre, noise type, and SNR.

Model labels used in the plots:

- `Base AudioSet embeddings`: embeddings from `630k-audioset-best.pt`
- `Fine-tuned Music embeddings`: embeddings from `music_audioset_epoch_15_esc_90.14.pt`

## Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate dataset overview plots:

```bash
python summarize_gtzan_data.py --data-root /path/to/embedding-root
```

Run genre classification evaluation:

```bash
python evaluate_gtzan_retrieval.py \
  --embedding-root /path/to/base-embeddings \
  --embedding-root /path/to/fine-tuned-embeddings \
  --model-label "Base AudioSet embeddings" \
  --model-label "Fine-tuned Music embeddings"
```

Run exact-song retrieval evaluation:

```bash
python evaluate_gtzan_exact_retrieval.py \
  --embedding-root /path/to/base-embeddings \
  --embedding-root /path/to/fine-tuned-embeddings \
  --model-label "Base AudioSet embeddings" \
  --model-label "Fine-tuned Music embeddings"
```

Add `--write-csv` to any script to write the underlying summary tables.

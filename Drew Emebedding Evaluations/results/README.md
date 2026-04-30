# Results

## Model Labels

- `Base AudioSet embeddings` = embeddings generated from `630k-audioset-best.pt`
- `Fine-tuned Music embeddings` = embeddings generated from `music_audioset_epoch_15_esc_90.14.pt`

These labels describe which model checkpoint created the embeddings. They are separate from the audio condition labels below.

## Audio Conditions

- `Clean original tracks` = files under `Data/genres_original`
- `Noisy augmented tracks` = files under `Data/genres_augmented`
- SNR levels: `0 dB`, `10 dB`, `20 dB`

## Folders

- `genre_classification/`: train on clean originals, predict genre for noisy augmented tracks, and compare to the GTZAN genre folder.
- `exact_song_retrieval/`: use noisy augmented tracks to retrieve from clean originals and compare to the clean track with the same `track_id`.
- `data_overview/`: basic dataset structure plots

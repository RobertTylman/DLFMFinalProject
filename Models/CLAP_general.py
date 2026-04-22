from pathlib import Path
import json
import numpy as np
import laion_clap


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# ===== HARDCODE YOUR PATHS HERE =====
input_root = Path("/Users/katelynvieni/Downloads/GTZAN Dataset")
output_root = Path("/Users/katelynvieni/Downloads/GTZAN_embeddings")

# Optional: set to a checkpoint path, or None to use default
ckpt_path = None
use_fusion = False
batch_size = 8
# ===================================


def collect_audio_files(input_root: Path):
    if not input_root.exists():
        raise ValueError(f"Input path does not exist: {input_root}")
    if not input_root.is_dir():
        raise ValueError(f"Expected a directory, got: {input_root}")

    files = sorted(
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )

    if not files:
        raise ValueError(f"No supported audio files found in: {input_root}")

    return files


def main():
    output_root.mkdir(parents=True, exist_ok=True)

    audio_files = collect_audio_files(input_root)
    print(f"Found {len(audio_files)} audio file(s).")

    model = laion_clap.CLAP_Module(enable_fusion=use_fusion)

    if ckpt_path:
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_ckpt(ckpt=str(ckpt_path))
    else:
        print("Loading default laion-clap checkpoint...")
        model.load_ckpt()

    manifest = []

    for start in range(0, len(audio_files), batch_size):
        batch = audio_files[start:start + batch_size]
        batch_paths = [str(p) for p in batch]

        print(f"Embedding files {start + 1}-{start + len(batch)} / {len(audio_files)}")
        embeds = model.get_audio_embedding_from_filelist(
            x=batch_paths,
            use_tensor=False
        )

        for audio_path, emb in zip(batch, embeds):
            rel_path = audio_path.relative_to(input_root)
            out_path = output_root / rel_path.with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(out_path, emb.astype(np.float32))

            manifest.append({
                "audio_path": str(audio_path),
                "relative_audio_path": str(rel_path),
                "embedding_path": str(out_path),
                "relative_embedding_path": str(out_path.relative_to(output_root)),
                "embedding_shape": list(emb.shape),
            })

    manifest_path = output_root / "embedding_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "input_root": str(input_root),
                "output_root": str(output_root),
                "checkpoint": str(ckpt_path) if ckpt_path else "default laion-clap non-fusion checkpoint",
                "fusion": use_fusion,
                "num_files": len(manifest),
                "items": manifest,
            },
            f,
            indent=2
        )

    print(f"Saved per-file embeddings under: {output_root}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
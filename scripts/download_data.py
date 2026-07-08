"""
================================================================================
  DATASET DOWNLOADER — RAVDESS from Zenodo  (no Kaggle login needed)
================================================================================
  Downloads the RAVDESS speech audio (~215 MB, 1440 files, 24 actors)
  directly from Zenodo and extracts it into data/RAVDESS/.

  Run once:
      python download_data.py

  After this, your folder structure will be:
      data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav
      data/RAVDESS/Actor_01/03-01-02-01-01-01-01.wav
      ...
      data/RAVDESS/Actor_24/...

  Alternative (if this script fails) — download manually:
      https://zenodo.org/records/1188976
      → click  "Audio_Speech_Actors_01-24.zip"  (215 MB)
      → unzip into data/RAVDESS/

  For TESS (2 speakers, smaller):
      https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
      (needs free Kaggle account; use `kaggle datasets download` CLI)
================================================================================
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"
TARGET_DIR  = Path("data/RAVDESS")
ZIP_PATH    = Path("data/ravdess_speech.zip")


def download(url, dest):
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, 100 * downloaded / total_size) if total_size else 0
        mb_done  = downloaded / 1e6
        mb_total = total_size / 1e6
        sys.stdout.write(f"\r  {mb_done:6.1f} / {mb_total:.1f} MB  ({pct:5.1f}%)")
        sys.stdout.flush()

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print("\n✓ Download complete.")


def main():
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)

    if TARGET_DIR.exists() and any(TARGET_DIR.rglob("*.wav")):
        print(f"✓ Data already present in {TARGET_DIR} — skipping download.")
        return

    if not ZIP_PATH.exists():
        download(RAVDESS_URL, ZIP_PATH)

    print(f"Extracting to {TARGET_DIR} ...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(TARGET_DIR)
    print("✓ Extraction complete.")

    n = sum(1 for _ in TARGET_DIR.rglob("*.wav"))
    print(f"✓ {n} .wav files ready in {TARGET_DIR}/")

    # optional cleanup
    try:
        ZIP_PATH.unlink()
        print("  (removed zip archive)")
    except OSError:
        pass


if __name__ == "__main__":
    main()

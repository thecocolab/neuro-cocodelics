import os
from glob import glob
from pathlib import Path

import mne
import numpy as np
from toto.inference.embedding import embed
from utils import batched_sliding_window

WINDOW_SECONDS = 30
STEP_SECONDS = 10
DATA_DIR = "data_sprint/processed"
OUT_DIR = f"data_sprint/embeddings-w{WINDOW_SECONDS}s-s{STEP_SECONDS}s"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for path in glob((Path(DATA_DIR) / "**" / "*.fif").as_posix(), recursive=True):
        dataset, subject, session, _ = Path(path).parts[-4:]
        raw = mne.io.read_raw_fif(path, preload=True)

        # compute embeddings
        _, embeddings = batched_sliding_window(
            raw, embed, window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS, batch_size=4
        )
        embeddings = np.array(embeddings)
        np.save(Path(OUT_DIR) / dataset / subject / session / f"embeddings.npy", embeddings)


if __name__ == "__main__":
    main()

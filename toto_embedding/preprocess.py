import os
from pathlib import Path

import mne
import pandas as pd
from joblib import Parallel, delayed
from mne_bids import BIDSPath
from tqdm.auto import trange

DATA_DIR = "data_sprint"
TGT_SFREQ = 256
OUT_DIR = "data_sprint/processed"


def process_file(row):
    bids_path = BIDSPath(
        subject=row.subject,
        session=row.session,
        task="rest",
        datatype="meg",
        root=DATA_DIR,
    )
    bids_path.update(suffix="meg", extension=".fif")
    raw = mne.io.read_raw_fif(row.path)

    # resample and filter
    raw.resample(TGT_SFREQ, n_jobs=-1, verbose=False)
    raw.filter(1, 100, n_jobs=-1, verbose=False)

    outpath = Path(OUT_DIR) / row.dataset / row.subject / row.session / "data_meg.fif"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    raw.save(outpath, overwrite=True, verbose=False)


def main():
    df = pd.DataFrame(columns=["dataset", "subject", "session", "path"])
    for path in Path(DATA_DIR).glob("MEG_*/**/sub-*.fif"):
        dset = path.parts[-6].replace("MEG_", "")
        subj = path.parts[-4].replace("sub-", "")
        sess = path.parts[-3].replace("ses-", "")
        df.loc[len(df)] = [dset, subj, sess, str(path)]

    Parallel(n_jobs=-1)(delayed(process_file)(df.iloc[i]) for i in trange(len(df), desc="Processing files"))
    print("Processing complete. Processed files saved to:", OUT_DIR)


if __name__ == "__main__":
    main()

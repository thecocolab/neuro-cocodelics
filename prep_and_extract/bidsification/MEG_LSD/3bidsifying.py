import pandas as pd
import os
import mne
from mne_bids import BIDSPath, write_raw_bids
from sovabids.parsers import parse_from_placeholder
import traceback

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Adjust width as needed

csv = os.path.join(os.path.dirname(__file__), 'meg_bids.csv')

df = pd.read_csv(csv)

BIDS_ROOT = '/home/yorguin/projects/rrg-kjerbi/shared/neuro-cocodelics/bids/MEG_LSD'
os.makedirs(BIDS_ROOT, exist_ok=True)

errors = []

for i, row in df.iterrows():
    subject = row['subject_label']
    session = row['session']
    task = row['task']
    filepath = row['file']

    try:
        bids_path = BIDSPath(subject=subject, session=session, task=task, root=BIDS_ROOT, datatype='meg', suffix='meg', extension='.fif')
        if not os.path.isfile(bids_path.fpath):
            mne_data = mne.io.read_raw(filepath, preload=True)
            write_raw_bids(mne_data, bids_path=bids_path, overwrite=True, format="FIF", allow_preload=True)
        else:
            print(f"File {bids_path.fpath} already exists, skipping.")
        print(f"Processed {i+1}/{len(df)}: {subject}, {session}, {task}, {filepath}")
    except Exception as e:
        print(f"Error processing {i+1}/{len(df)}: {subject}, {session}, {task}, {filepath}")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        errors.append({
            'index': i,
            'subject': subject,
            'session': session,
            'task': task,
            'filepath': filepath,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        continue

# Save errors to a CSV file for later inspection
if errors:
    errors_df = pd.DataFrame(errors)
    errors_csv = os.path.join(BIDS_ROOT, 'bids_errors.csv')
    errors_df.to_csv(errors_csv, index=False)
    print(f"Errors saved to {errors_csv}")

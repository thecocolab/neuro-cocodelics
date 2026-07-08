import pandas as pd
import os
import mne
from mne_bids import BIDSPath, write_raw_bids
from sovabids.parsers import parse_from_placeholder
#import mat73
#import mne
import numpy as np
import glob
import scipy.io as sio
import traceback
import os
from pprint import pprint
import traceback

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Adjust width as needed



datasets = {
#    'MEG_ketamine': '/home/yorguin/projects/def-kjerbi/data/MEG_ketamine/',
    'MEG_perampanel': '/home/yorguin/projects/def-kjerbi/data/MEG_perampanel/',
#    'MEG_psilocybin':'/home/yorguin/projects/def-kjerbi/data/MEG_psilocybin/',
#    'MEG_tiagabine': '/home/yorguin/projects/def-kjerbi/data/MEG_tiagabine/',
}

OUTPUT_PATH = '/home/yorguin/projects/rrg-kjerbi/shared/neuro-cocodelics/MEG_perampanel/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

def loadmat(x,kwargs={}):
    print(f"Loading {x} with scipy.io.loadmat")
    return sio.loadmat(x,**kwargs)


def get_metadata(meg, filepath=None):
    try:
        #kwargs={'squeeze_me': True, 'struct_as_record': False})
        #squeeze_me=True, struct_as_record=False)
        data_dict = meg['data']  # Assuming 'data' is the key for the main data structure
        # 1. Extract data
        fsample = data_dict['fsample']  # e.g., 1200
        labels = data_dict['label']     # list of channel names
        data_shape = data_dict['trial'].shape    # assuming continuous → one array of shape (n_channels, n_samples)
        times = data_dict['time']    # should be 1D array of length n_samples
        try:
            event = data_dict['cfg']['event']  # Dict with keys: 'duration', 'offset', 'sample', 'type', 'value'
            trl = data_dict['cfg']['trl']      # Usually a numpy array with shape (n_trials, 3)
        except:
            event = None
            trl = None
        try:
            cfg_keys = list(data_dict['cfg'].keys())  # Get all keys in the cfg dictionary
        except:
            cfg_keys = None
        # make dictionary with this information

        metadata = {
            'filepath': filepath if filepath else 'not provided',
            'fsample': fsample,
            'labels': labels,
            'data_shape': data_shape,
            'times': times,
            'event': event,
            'trl': trl,
            'cfg_keys': cfg_keys
        }
        return metadata
    except Exception as e:
        metadata = {
            'filepath': filepath if filepath else 'not provided',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        return metadata


def inspect_meg_data(mats, file='meg_data.txt'):
    amount = len(mats)
    metadatas = []
    for i, meg_path in enumerate(mats):
        print(f"Processing {i+1}/{amount}: {meg_path}")
        meg_data = loadmat(meg_path, kwargs=dict(simplify_cells=True))
        metadata = get_metadata(meg_data, meg_path)
        with open(file, 'a') as f:
            pprint(metadata, stream=f)
        metadatas.append(metadata)
    return metadatas


filepath = os.path.join(OUTPUT_PATH, 'meg_perampanel_metadata.csv')
filepath_pkl = os.path.join(OUTPUT_PATH, 'meg_perampanel_metadata.pkl')

if not os.path.exists(filepath):
    print(f"File {filepath} does not exist, inspecting datasets...")
    FILES_PER_DATASET = None  # Number of files to inspect per dataset
    metadatas_dict = {}
    for dataset_name, dataset_path in datasets.items():
        print(f"Inspecting dataset: {dataset_name} at {dataset_path}")
        mats = glob.glob(os.path.join(dataset_path, '**', '*.mat'), recursive=True)
        root = os.path.dirname(dataset_path)

        print(f"Found {len(mats)} .mat files in {dataset_path}, limiting to {FILES_PER_DATASET} files for inspection.")

        if FILES_PER_DATASET is not None:
            mats = mats[:FILES_PER_DATASET]  if len(mats) > FILES_PER_DATASET else mats

        metadatas_dict[dataset_name] = []
        filepath = os.path.join(OUTPUT_PATH, f'{dataset_name}_meg_data.txt')
        metadatas = inspect_meg_data(mats, file=filepath)
        metadatas_dict[dataset_name] = metadatas
        print(f"Inspection results saved to {dataset_name}_meg_data.txt")

    df = pd.DataFrame.from_records(metadatas_dict['MEG_perampanel'])

    df.to_csv(filepath, index=False, sep=';', encoding='utf-8')
    df.to_pickle(filepath_pkl)
else:
    print(f"File {filepath} already exists, loading metadata from CSV...")
    df = pd.read_csv(filepath, sep=';', encoding='utf-8')

df['filepath'].iloc[0]
df['data_shape']
df['fsample'].value_counts()
df.columns
#df['event'].apply(lambda x: len(x) if x is not None else 0).value_counts()
df['event'].iloc[0]

for i, row in df.iterrows():
    print(i,row['event'])


# Inspect errors
errors  = df[pd.isna(df['fsample'])]

df['cfg_keys'].iloc[0]

for i, row in errors.iterrows():
    print(row['filepath'])
    print(row['error'])
    print(row['traceback'])

    # meg_data = loadmat(row['filepath'], kwargs=dict(simplify_cells=True))
    # data = meg_data.get('data', None)  # Check if 'data' key exists
    # data.keys()
    
    # data['trial'].shape
    # data['label']
    # data['time']
    # data['cfg'].keys()
    # data['cfg']['trials']


# '/home/yorguin/scratch/data/MEG_perampanel/meg_data/PMP_PMP_020414_50.mat'
pattern = r'/home/yorguin/projects/def-kjerbi/data/MEG_perampanel/meg_data/PMP_%session%_%subject%_%number%.mat'

bids_items = []
for i, row in df.iterrows():
    bids_dict = parse_from_placeholder(row['filepath'],pattern)
    filepath = row['filepath']
    bids_dict['filepath'] = filepath
    bids_items.append(bids_dict)


df_bids = pd.DataFrame(bids_items)

df_bids['subject'].unique().shape

df_bids = df_bids.sort_values(by=['subject', 'session'])

df_bids['filename'] = df_bids['filepath'].apply(lambda x: os.path.basename(x))

df_bids['label'] = 'S' + df_bids['subject'] + 'N' + df_bids['number']

df_bids['session_bids'] = df_bids['session'].apply(lambda x: 'placebo' if x == 'PLA' else 'perampanel')



# Most of them have 2 events, or 0.
# Will convert as is, to a raw file

BIDS_ROOT = '/home/yorguin/projects/rrg-kjerbi/shared/neuro-cocodelics/MEG_perampanel'
os.makedirs(BIDS_ROOT, exist_ok=True)
errors = []
for i, row in df_bids.iterrows():
    try:
        print(f"Processing {i+1}/{len(df)}: {row['filepath']}")
        print(row)
        meg_path = row['filepath']
        meg_data = loadmat(meg_path, kwargs=dict(simplify_cells=True))
        metadata = get_metadata(meg_data, meg_path)

        chantype_map = {
        'meggrad': 'grad',     # MNE type for gradiometers
        'refgrad': 'grad',     # if you want to treat refgrad as grad
        'refmag': 'mag'        # or maybe 'misc' if they are not standard MEG
        }


        ch_names = meg_data['data']['label'].tolist()
        types = [chantype_map.get(ch, 'misc') for ch in ch_names]  # Default to 'misc' if not found
        chantypes = ['meggrad', 'meggrad', 'refmag', 'refgrad']
        mne_types = [chantype_map.get(t, 'misc') for t in chantypes]

        # Create a dictionary to pass to set_channel_types
        ch_type_dict = {name: typ for name, typ in zip(ch_names, mne_types)}

        # Set types
        raw = mne.io.RawArray(
            data=meg_data['data']['trial'],  # Assuming 'trial' is a 2D array with shape (n_channels, n_samples)
            info=mne.create_info(
                ch_names=meg_data['data']['label'].tolist(),  # Assuming 'label' is a list of channel names
                sfreq=meg_data['data']['fsample'],  # Assuming 'fsample' is a scalar
                ch_types=types  # Assuming all channels are meg
            )
        )

        raw.set_channel_types(ch_type_dict)

        subject = row['label']
        session = row['session_bids']
        task = 'resting'
        filepath = row['filepath']
        bids_path = BIDSPath(subject=subject, session=session, task=task, root=BIDS_ROOT, datatype='meg', suffix='meg', extension='.fif')
        if not os.path.isfile(bids_path.fpath):
            write_raw_bids(raw, bids_path=bids_path, overwrite=True, format="FIF", allow_preload=True)
        else:
            print(f"File {bids_path.fpath} already exists, skipping.")
    except Exception as e:
        print(f"Error processing {i+1}/{len(df)}: {row['filepath']}")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        errors.append({
            'index': i,
            'subject': row['subject'],
            'session': row['session'],
            'filepath': row['filepath'],
            'error': str(e),
            'traceback': traceback.format_exc()
        })

# Save errors to a CSV file for later inspection
if errors:
    errors_df = pd.DataFrame(errors)
    errors_csv = os.path.join(BIDS_ROOT, 'bids_errors.csv')
    errors_df.to_csv(errors_csv, index=False)


"""
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Adjust width as needed

csv = '/home/yorguin/projects/def-kjerbi/yorguin/datasets/MEG_LSD/meg_bids.csv'

df = pd.read_csv(csv)


for i, row in df.iterrows():

    mne_data = mne.io.read_raw(filepath, preload=True)
    bids_path = BIDSPath(subject=subject, session=session, task=task, root=BIDS_ROOT)
    write_raw_bids(mne_data, bids_path=bids_path, overwrite=True, format="FIF", allow_preload=True)
    print(f"Processed {i+1}/{len(df)}: {subject}, {session}, {task}, {filepath}")
"""


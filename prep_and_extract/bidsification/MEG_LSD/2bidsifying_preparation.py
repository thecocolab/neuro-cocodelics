import os
import pandas as pd
import numpy as np
import glob
from sovabids.parsers import parse_from_placeholder

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Adjust width as needed


ID_file = '/home/yorguin/projects/def-kjerbi/yorguin/datasets/MEG_LSD/IDs.xlsx'

df_ids = pd.read_excel(ID_file, header=2)

# drop nans
df_ids = df_ids.dropna(subset=['ID'])

# pass _ to -

df_ids['ID'] = df_ids['ID'].str.replace('_', '-', regex=False)

rename = {
    'Unnamed: 0':'Unnamed',
    'ID':'ID',
    'subject':'number',
    'Unnamed: 3':'label',
}

df_ids = df_ids.rename(columns=rename)

SOURCE_PATH = "/home/yorguin/scratch/data/MEG_LSDV2/meg_data"

folders = glob.glob(os.path.join(SOURCE_PATH, '*'), recursive=False)

# count the number of empty folders

empty_folders = [folder for folder in folders if os.path.isdir(folder) and not os.listdir(folder)]
non_empty_folders = [folder for folder in folders if os.path.isdir(folder) and os.listdir(folder)]

print(f"Number of empty folders: {len(empty_folders)}")
print(f"Number of non-empty folders: {len(non_empty_folders)}")


ds_list = glob.glob(os.path.join(SOURCE_PATH, '*.ds'), recursive=False)
megs = [ds for ds in ds_list if os.path.isdir(ds) and True] # Filter here if needed


# '/home/yorguin/scratch/data/MEG_LSDV2/meg_data/310714-1_LSD_20140917_Closed1.ds/hz.ds'
# '/home/yorguin/scratch/data/MEG_LSDV2/meg_data/1_LSD_20140626_01.ds'
pattern = r'/home/yorguin/scratch/data/MEG_LSDV2/meg_data/%subjectNumber%_LSD_%session%_%task%.ds'

megs_parsed = []
for meg_file in megs:
    try:
        bids_dict = parse_from_placeholder(meg_file,pattern=pattern)
        bids_dict['file']= meg_file  # Add the original file path to the BIDSPath
        #meg = mne.io.read_raw(meg_file, preload=True)  # Load the raw data
        print(f"Parsed BIDS path for {meg_file}: {bids_dict}")
        #print(meg)
        megs_parsed.append(bids_dict)
    except Exception as e:
        print(f"Error parsing BIDS path for {meg_file}: {e}")
#    break

# Convert the list of dictionaries to a DataFrame
df_megs = pd.DataFrame(megs_parsed)

# for i, row in df_megs.iterrows():
#     print(row['file'], row['subjectNumber'], row['session'], row['task'])


# Simplify the path for easier inspection
def simplify_path(path):
    """Simplify the path by removing the base directory."""
    base_dir = '/home/yorguin/scratch/data/MEG_LSDV2/meg_data/'
    return path.replace(base_dir, '')

df_megs['file'] = df_megs['file'].apply(simplify_path)


# Drop all MMN tasks
df_megs = df_megs[~df_megs['task'].str.contains('MMN', na=False)]


# Correct id as the previous parse_bids gets rid of "-" (non bids compliant)
def correct_id(id_str):
    if len(id_str) == 7:
        return id_str[:6] + '-' + id_str[6]
    else:
        return id_str

df_megs['subjectNumber'] = df_megs['subjectNumber'].apply(correct_id)



###########################################################################
# JORDAN NOTES


id_key = {
                '1':  '230911-1', 
                '2':  '010514-1', 
                '3':  '140514-1', 
                '4':  '140514-2', 
                '5':  '050913-1', 
                '6':  '290514-2', 
                '7':  '270613-1',
                # no 8
                '9':  '260614-1', 
                '10': '240107-6',
                '11': '090714-1', 
                '12': '090714-2',
                '13': '310714-1',
                '14': '041213-1', 
                '15': '070814-1',
                '16': '070814-2',
                '17': '200814-1', 
                '18': '010813-4', 
                '19': '040914-1', 
                '20': '040914-2', 
                }
pla_key = {
            '1':  ('20140515', "# on disk. 'Closed2' was named simply 'Closed'."),
            '2':  ('20140501', "# on disk"),
            '3':  ('20140528', "# on disk"),
            '4':  ('20140514', "# id was misnamed 280212-2 in ds"),
            '5':  ('20140612', "# on disk"),
            '6':  ('20140529', "# on disk"),
            '7':  ('20140625', "# on disk. 'Opoen1' instead of 'Open1'. Music missing."),
            '9':  ('20140710', "# on disk"),
            '10': ('20140626', "# on disk"),
            '11': ('20140730', "# on disk"),
            '12': ('20140709', "# id was misnamed 070914-2 in ds"),
            '13': ('20140917', "# on disk"),
            '14': ('20140731', "# on disk. 'Music' was saved as 'Audio'"),
            '15': ('20140821', "# on disk. 'Closed1' was named 'Closed'"),
            '16': ('20140807', "# on disk"),
            '17': ('20140903', "# on disk"),
            '18': ('20140820', "# on disk"),
            '19': ('20140918', "# on disk"),
            '20': ('20140904', "# on disk"),
            }
lsd_key = {
            '1':  ('20140501', "# id was misnamed 231109-1. There is a second music rec named 'Music2'."),
            '2':  ('20140515', "# id was misnamed 300414-1 in ds ('LSD Analysis.xlsx' misreports this mistake)"),
            '3':  ('20140514', "# on disk. There is a second music rec named 'Music2'."),
            '4':  ('20140528', "# on disk"),
            '5':  ('20140529', "# on disk. 'Open1' was saved as 'Open'"),
            '6':  ('20140612', "# on disk"),
            '7':  ('20140611', "# on disk. Music missing. According to Venkatesh, nothing to do here..."),
            '9':  ('20140626', "# on disk"),
            '10': ('20140710', "# on disk"),
            '11': ('20140709', "# on disk"),
            '12': ('20140730', "# on disk"),
            '13': ('20140731', "# on disk"),
            '14': ('20140917', "# on disk"),
            '15': ('20140807', "# on disk. There is a second Video named 'Video2'. Venkatesh says to use 'Video2' instead of Video (Video gives hpi info missing error)"),
            '16': ('20140821', "# on disk"),
            '17': ('20140820', "# on disk"),
            '18': ('20140903', "# id was misnamed 010814-4 in ds"),
            '19': ('20140904', "# on disk (in Supp)"),
            '20': ('20140918', "# on disk (in Supp)"),
            }

jordan_notes = {}

for key in id_key.keys():
    jordan_notes[key] = {}
    jordan_notes[key]['id'] = id_key[key]
    jordan_notes[key]['pla'] = pla_key[key]
    jordan_notes[key]['lsd'] = lsd_key[key]

jordan_df = pd.DataFrame.from_dict(jordan_notes, orient='index')


###########################################################################


# This is an example of a correct subject, there should be Video, Open1, Open2, Closed1, Closed2, Music, MMN1, MMN2, MMN3 tasks (9 tasks) (6 if MMN tasks are not included)
# 2 sessions: 1 Placebo, 1 LSD (so 9x2 = 18 files) per subject
correct_tasks = ['Video', 'Open1', 'Open2', 'Closed1', 'Closed2', 'Music'] #, 'MMN1', 'MMN2', 'MMN3']

def check_subject_tasks(subject_number, df,correct_tasks=correct_tasks,print_df=False):
    good = True
    df2 = df.copy()
    query_df = df2[df2['subjectNumber'] == subject_number].sort_values(by='task', ascending=False)
    #print(query_df)

    for task in query_df['task'].unique():
        if task not in correct_tasks:
            print(f"Subject {subject_number} has an unexpected task: {task}")
            good = False

    for task in correct_tasks:
        if task not in query_df['task'].values:
            print(f"Subject {subject_number} is missing task: {task}")
            good = False
    
    for task in correct_tasks:
        query2_df_task = query_df.copy()[query_df['task'] == task]
        if len(query2_df_task) != 2:
            print(f"Subject {subject_number} has {len(query2_df_task)} files for task {task}, expected 2.")
            good = False

    if good:
        print(f"Subject {subject_number} has exactly expected tasks and files.")
    else:
        print(f"Subject {subject_number} has issues with tasks.")

    if print_df:
        print(query_df)
    return good,query_df


##### Example of a correct subject ########
subject = '200814-1'
check_subject_tasks(subject,df_megs)

#####################


def check_all_subjects(df, correct_tasks=correct_tasks):
    df_ = df.copy()
    bad_subjects = []
    good_subjects = []
    for subject in df_['subjectNumber'].unique().tolist():
        print(f"Checking subject {subject}")
        good,query_df = check_subject_tasks(subject, df_.copy(),correct_tasks=correct_tasks, print_df=False)
        if not good:
            bad_subjects.append(subject)
        else:
            print(f"Subject {subject} is good.")
            good_subjects.append(subject)
        print('#'*50)
    return bad_subjects, good_subjects

check_all_subjects(df_megs, correct_tasks=correct_tasks)
#####################################################################

# Drop unexpected subjects : TEST, 1,2.

unexpected_subjects = ['TEST', '1', '2', '23041451']

unused_subjects_for_some_reason = ['140614-1', '300114-1', '071012-1']
unexpected_subjects.extend(unused_subjects_for_some_reason)

df_megs = df_megs[~df_megs['subjectNumber'].isin(unexpected_subjects)]

####################################################################
# Make corrections based on Jordan's notes

df_megs =  df_megs.sort_values(by=['subjectNumber'], ascending=True)
df_megs_corrected = df_megs.copy()

###################################################################

def placebo_lsd_logic(df,jordan_df, this_subject):
    df_megs_corrected = df.copy()
    # Now apply session placebo/lsd logic

    placebo_session = jordan_df[jordan_df['id'] == this_subject]['pla'].values[0][0]
    lsd_session = jordan_df[jordan_df['id'] == this_subject]['lsd'].values[0][0]

    df_megs_corrected.loc[(df_megs_corrected['subjectNumber'] == this_subject) & (df_megs_corrected['session'] == placebo_session), 'session'] = 'placebo'
    df_megs_corrected.loc[(df_megs_corrected['subjectNumber'] == this_subject) & (df_megs_corrected['session'] == lsd_session), 'session'] = 'lsd'

    print(df_megs_corrected[df_megs_corrected['subjectNumber'] == this_subject])

    check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)
    return df_megs_corrected

def print_jordan_notes(this_subject):
    notes_row = jordan_df[jordan_df['id'] == this_subject].iloc[0]
    for col in notes_row.index:
        print(f"{col}: {notes_row[col]}")

####################################################################

# Apply placebo/lsd logic to initial good megs

initial_bads,initial_goods = check_all_subjects(df_megs, correct_tasks=correct_tasks)

for this_subject in initial_goods:
    print(f"Applying placebo/lsd logic to subject {this_subject}")
    df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

####################################################################

# 070814-1
this_subject = '070814-1'
# Subject 070814-1 has an unexpected task: Video2
# Subject 070814-1 has an unexpected task: Closed
# Subject 070814-1 has 1 files for task Closed1, expected 2.
# Subject 070814-1 has issues with tasks.

check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=True)

print_jordan_notes(this_subject)
# id: 070814-1
# pla: ('20140821', "# on disk. 'Closed1' was named 'Closed'")
# lsd: ('20140807', "# on disk. There is a second Video named 'Video2'.")

index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['session'] == '20140821') & (df_megs['task'] == 'Closed')].index
df_megs_corrected.loc[index, 'task'] = 'Closed1'


# Actually, according to Venkatesh, use 'Video2' instead of Video (Video gives hpi info missing error)

index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['session'] == '20140807') & (df_megs['task'] == 'Video')].index

df_megs_corrected = df_megs_corrected.drop(index)

index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['session'] == '20140807') & (df_megs['task'] == 'Video2')].index

# rename Video2 to Video
df_megs_corrected.loc[index, 'task'] = 'Video'

df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

##################################################
# Subject 050913-1 has an unexpected task: Open
# Subject 050913-1 has 1 files for task Open1, expected 2.
# Subject 050913-1 has issues with tasks.

this_subject = '050913-1'
check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)

df_megs[df_megs['subjectNumber'] == this_subject]
# id: 050913-1
# pla: ('20140612', '# on disk')
# lsd: ('20140529', "# on disk. 'Open1' was saved as 'Open'")

index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['session'] == '20140529') & (df_megs['task'] == 'Open')].index
df_megs_corrected.loc[index, 'task'] = 'Open1'


# Now apply session placebo/lsd logic
df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

#######################################################################
# Checking subject 240107-6
# Subject 240107-6 has an unexpected task: Closed12
# Subject 240107-6 has issues with tasks.

this_subject = '240107-6'

print_jordan_notes(this_subject)

check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)
# drop the task Closed12
index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['task'] == 'Closed12')].index

df_megs_corrected = df_megs_corrected.drop(index)

#now apply session placebo/lsd logic

df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)


########################################################################
# Checking subject 041213-1
# Subject 041213-1 has an unexpected task: Audio
# Subject 041213-1 has 1 files for task Music, expected 2.
# Subject 041213-1 has issues with tasks.

this_subject = '041213-1'
print_jordan_notes(this_subject)

check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)

index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['task'] == 'Audio')].index

# replace Audio with Music
df_megs_corrected.loc[index, 'task'] = 'Music'

# Now apply session placebo/lsd logic

df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

########################################################################

# Checking subject 230911-1
this_subject = '230911-1'
print_jordan_notes(this_subject)
# id: 230911-1
# pla: ('20140515', "# on disk. 'Closed2' was named simply 'Closed'.")
# lsd: ('20140501', "# id was misnamed 231109-1. There is a second music rec named 'Music2'.")

check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)

other_alias = '231109-1'

check_subject_tasks(other_alias, df_megs, correct_tasks=correct_tasks, print_df=False)

# rename the subjectNumber to the correct id

indexes = df_megs_corrected[df_megs_corrected['subjectNumber'] == other_alias].index
df_megs_corrected.loc[indexes, 'subjectNumber'] = this_subject

check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

index = df_megs_corrected[(df_megs_corrected['subjectNumber'] == this_subject) & (df_megs_corrected['task'] == 'Music2')].index
df_megs_corrected = df_megs_corrected.drop(index)

index = df_megs_corrected[(df_megs_corrected['subjectNumber'] == this_subject) & (df_megs_corrected['session'] == '20140515') & (df_megs_corrected['task'] == 'Closed')].index
df_megs_corrected.loc[index, 'task'] = 'Closed2'

# Now apply session placebo/lsd logic
df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)
#############################################

# Checking subject 010514-1
# Subject 010514-1 has 1 files for task Video, expected 2.
# Subject 010514-1 has 1 files for task Open1, expected 2.
# Subject 010514-1 has 1 files for task Open2, expected 2.
# Subject 010514-1 has 1 files for task Closed1, expected 2.
# Subject 010514-1 has 1 files for task Closed2, expected 2.
# Subject 010514-1 has 1 files for task Music, expected 2.
# Subject 010514-1 has issues with tasks.

this_subject = '010514-1'
print_jordan_notes(this_subject)
# id: 010514-1
# pla: ('20140501', '# on disk')
# lsd: ('20140515', "# id was misnamed 300414-1 in ds ('LSD Analysis.xlsx' misreports this mistake)")

check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)

other_alias = '300414-1'
check_subject_tasks(other_alias, df_megs, correct_tasks=correct_tasks, print_df=False)

# rename the subjectNumber to the correct id
indexes = df_megs_corrected[df_megs_corrected['subjectNumber'] == other_alias].index
df_megs_corrected.loc[indexes, 'subjectNumber'] = this_subject

check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

# now apply session placebo/lsd logic

df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)


##################################################################################
# Checking subject 140514-1
# Subject 140514-1 has an unexpected task: Music2
# Subject 140514-1 has issues with tasks.

this_subject = '140514-1'
print_jordan_notes(this_subject)

# id: 140514-1
# pla: ('20140528', '# on disk')
# lsd: ('20140514', "# on disk. There is a second music rec named 'Music2'.")

check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)
index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['task'] == 'Music2')].index
df_megs_corrected = df_megs_corrected.drop(index)

# Now apply session placebo/lsd logic
df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

##################################################################################

# Checking subject 090714-2
# Subject 090714-2 has 1 files for task Video, expected 2.
# Subject 090714-2 has 1 files for task Open1, expected 2.
# Subject 090714-2 has 1 files for task Open2, expected 2.
# Subject 090714-2 has 1 files for task Closed1, expected 2.
# Subject 090714-2 has 1 files for task Closed2, expected 2.
# Subject 090714-2 has 1 files for task Music, expected 2.
# Subject 090714-2 has issues with tasks.

this_subject = '090714-2'
print_jordan_notes(this_subject)

# id: 090714-2
# pla: ('20140709', '# id was misnamed 070914-2 in ds')
# lsd: ('20140730', '# on disk')

check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)
other_alias = '070914-2'
check_subject_tasks(other_alias, df_megs, correct_tasks=correct_tasks, print_df=False)
# rename the subjectNumber to the correct id
indexes = df_megs_corrected[df_megs_corrected['subjectNumber'] == other_alias].index
df_megs_corrected.loc[indexes, 'subjectNumber'] = this_subject
check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

# now apply session placebo/lsd logic
df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

##################################################################################

# Checking subject 010813-4
# Subject 010813-4 has 1 files for task Video, expected 2.
# Subject 010813-4 has 1 files for task Open1, expected 2.
# Subject 010813-4 has 1 files for task Open2, expected 2.
# Subject 010813-4 has 1 files for task Closed1, expected 2.
# Subject 010813-4 has 1 files for task Closed2, expected 2.
# Subject 010813-4 has 1 files for task Music, expected 2.
# Subject 010813-4 has issues with tasks.

this_subject = '010813-4'
print_jordan_notes(this_subject)

# id: 010813-4
# pla: ('20140820', '# on disk')
# lsd: ('20140903', '# id was misnamed 010814-4 in ds')

check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)

other_alias = '010814-4'

check_subject_tasks(other_alias, df_megs, correct_tasks=correct_tasks, print_df=False)

# rename the subjectNumber to the correct id
indexes = df_megs_corrected[df_megs_corrected['subjectNumber'] == other_alias].index
df_megs_corrected.loc[indexes, 'subjectNumber'] = this_subject
check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

# now apply session placebo/lsd logic
df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

##################################################################################
# Checking subject 140514-2
# Subject 140514-2 has 1 files for task Video, expected 2.
# Subject 140514-2 has 1 files for task Open1, expected 2.
# Subject 140514-2 has 1 files for task Open2, expected 2.
# Subject 140514-2 has 1 files for task Closed1, expected 2.
# Subject 140514-2 has 1 files for task Closed2, expected 2.
# Subject 140514-2 has 1 files for task Music, expected 2.
# Subject 140514-2 has issues with tasks.

this_subject = '140514-2'
print_jordan_notes(this_subject)

# id: 140514-2
# pla: ('20140514', '# id was misnamed 280212-2 in ds')
# lsd: ('20140528', '# on disk')

check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)
other_alias = '280212-2'
check_subject_tasks(other_alias, df_megs, correct_tasks=correct_tasks, print_df=False)

# rename the subjectNumber to the correct id
indexes = df_megs_corrected[df_megs_corrected['subjectNumber'] == other_alias].index
df_megs_corrected.loc[indexes, 'subjectNumber'] = this_subject
check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

# now apply session placebo/lsd logic
df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)
##################################################################################

# Checking subject 270613-1
# Subject 270613-1 has an unexpected task: Opoen1
# Subject 270613-1 is missing task: Music
# Subject 270613-1 has 1 files for task Open1, expected 2.
# Subject 270613-1 has 0 files for task Music, expected 2.
# Subject 270613-1 has issues with tasks.

this_subject = '270613-1'
print_jordan_notes(this_subject)

# id: 270613-1
# pla: ('20140625', "# on disk. 'Opoen1' instead of 'Open1'. Music missing.")
# lsd: ('20140611', '# on disk. Music missing.')

check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)
index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['task'] == 'Opoen1')].index
df_megs_corrected.loc[index, 'task'] = 'Open1'
# Now apply session placebo/lsd logic
df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

# Missing music task expected...


bads,_=check_all_subjects(df_megs_corrected, correct_tasks=correct_tasks)

# set(bads).intersection(set(list(jordan_df['id'].values)))
# that was used to identify unexpected subjects (outside of jordan notes)

print(f"Bad subjects: {bads}")

# restore fullpaths
df_megs_corrected['file'] = df_megs_corrected['file'].apply(lambda x: os.path.join(SOURCE_PATH, x))

df_megs = df_megs_corrected.copy()


def valid_subject(subject_number, df_ids=df_ids):
    return subject_number in df_ids['ID'].values

df_megs['valid_subject'] = df_megs['subjectNumber'].apply(valid_subject)

assert df_megs['valid_subject'].all(), "There are invalid subjects in the DataFrame."

def get_subject_label(subject_number, df_ids=df_ids):
    if subject_number in df_ids['ID'].values:
        return df_ids[df_ids['ID'] == subject_number]['label'].values[0]
    else:
        return False

df_megs['subject_label'] = df_megs['subjectNumber'].apply(get_subject_label)

df_megs.to_csv('/home/yorguin/projects/def-kjerbi/yorguin/datasets/MEG_LSD/meg_bids.csv', index=False)
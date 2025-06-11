"""
Scripts that cleans an aggregated CSV file and splits it into individual dataset-specific CSV files.

This script performs the following operations:
1. Loads the aggregate CSV file (".local_data/aggregate@raw.csv") containing all features from multiple datasets.
2. Filters out rows where the 'filepath' column contains 'split-02' or 'split-03', and removes the '_split-01' substring from remaining rows.
3. Checks for any NaN values in the dataset:
    - Identifies and logs details about columns and rows with NaN values.
    - Drops rows with NaN values.
4. Extracts and logs unique features and sensors based on the naming convention in the column names.
5. Verifies that each dataset contains the same features and sensors; logs warnings if inconsistencies are found.
6. Calculates and logs the number of unique subjects per dataset.
7. Creates a unique identifier for each subject by concatenating the dataset name with the subject and then factorizing it into unique integers.
8. Generates a binary target variable based on the presence of 'ses-placebo' in the 'filepath'.
9. For the LSD dataset:
    - Splits it into task-specific datasets by appending the task name to the dataset identifier.
    - Aggregates all LSD tasks into an 'lsd-avg' dataset by averaging numeric columns.
10. Drops unnecessary columns ('filepath', 'id', 'task') and reorders the columns to ensure 'subject' is the first column.
11. Logs the overall cleaning and splitting process, including the final shapes of dataframes.
12. Saves the cleaned aggregate dataframe and the individual dataset CSV files into the ".local_data" directory.
Clean up aggregate@raw.csv, the csv file containing all the features from all the 
datasets, and split it into multiple csv files, one for each dataset.
"""
import os
import pandas as pd
import warnings
import logging
from datetime import datetime

# Ensure logs directory exists
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"split_csv_{timestamp}.log")

# Configure logging to output messages to both console and a log file
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

if __name__ == "__main__":

    # Load the aggregate CSV file
    aggregate_file = "./local_data/aggregate@raw.csv"
    if not os.path.exists(aggregate_file):
        raise FileNotFoundError(f"Aggregate file {aggregate_file} does not exist.")
    aggregate_df = pd.read_csv(aggregate_file)

    logging.info(f"Loaded aggregate file with {len(aggregate_df)} rows and {len(aggregate_df.columns)} columns.")

    # drop rows that has their filepath has split-02 or split-03 in the name and remove split-01 from the name
    aggregate_df = aggregate_df[~aggregate_df['filepath'].str.contains('split-02|split-03')].copy()
    aggregate_df.loc[:, 'filepath'] = aggregate_df['filepath'].str.replace('_split-01', '', regex=False)

    logging.info(f"After filtering, {len(aggregate_df)} rows remain.")

    # check if dataset has nan or empty values
    if aggregate_df.isnull().values.any():
        nan_columns = aggregate_df.columns[aggregate_df.isnull().any()].tolist()
        nan_rows = {col: aggregate_df[aggregate_df[col].isnull()].index.tolist() for col in nan_columns}
        unique_nan_rows = list(set().union(*nan_rows.values()))
        unique_nan_features = set(col.split('.spaces')[0] for col in nan_columns if '.spaces' in col)
        unique_nan_sensors = set(col.split('.spaces')[1] for col in nan_columns if '.spaces' in col)
        nan_filepaths = list(set(aggregate_df.loc[list(unique_nan_rows), 'filepath'].unique().tolist()))

        # Drop rows with NaN values
        aggregate_df.dropna(inplace=True)
        logging.warning(f"Found NaN values in the dataset. These rows have been dropped."
                        f"Rows with NaN values: {unique_nan_rows}. "
                        f"Unique sensors with NaN values: {unique_nan_sensors}. "
                        f"Unique features with NaN values: {unique_nan_features}. "
                        f"Filepaths of these rows with NaN values: {nan_filepaths}")

    datasets = aggregate_df['dataset'].unique()
    logging.info(f"Found {len(datasets)} datasets: {datasets}")

    unique_features = set(col.split('.spaces-')[0] for col in aggregate_df.columns if '.spaces-' in col)
    unique_features = {feature.replace('feature-', '') for feature in unique_features}
    unique_sensors = set(col.split('.spaces-')[1] for col in aggregate_df.columns if '.spaces-' in col)

    logging.info(f"Found {len(unique_features)} unique features: {unique_features}")
    logging.info(f"Found {len(unique_sensors)} unique sensors: {unique_sensors}")

    # Make sure all datasets have the same features and sensors
    for dataset in datasets:
        dataset_df = aggregate_df[aggregate_df['dataset'] == dataset]
        dataset_features = set(col.split('.spaces-')[0] for col in dataset_df.columns if '.spaces-' in col)
        dataset_features = {feature.replace('feature-', '') for feature in dataset_features}
        dataset_sensors = set(col.split('.spaces-')[1] for col in dataset_df.columns if '.spaces-' in col)

        logging.info(f"Dataset {dataset} has {len(dataset_features)} features.")
        logging.info(f"Dataset {dataset} has {len(dataset_sensors)} sensors.")
        if dataset_features != unique_features:
            logging.warning(f"Dataset {dataset} has different features: {dataset_features - unique_features}")
        if dataset_sensors != unique_sensors:
            logging.warning(f"Dataset {dataset} has different sensors: {dataset_sensors - unique_sensors}")

    # Get number of UNIQUE subjects in each dataset
    subjects_per_dataset = aggregate_df.groupby('dataset')['subject'].nunique().to_dict()
    logging.info(f"Number of unique subjects per dataset: {subjects_per_dataset}")

    # Some IDs are not unique across datasets, so we need to create a unique identifier for each subject
    aggregate_df['subject'] = aggregate_df['dataset'] + '_' + aggregate_df['subject']
    # Convert subject column to unique integer identifiers
    aggregate_df['subject'] = pd.factorize(aggregate_df['subject'])[0] + 1
    logging.info("Converted subject column to unique integer identifiers across all datasets.")

    aggregate_df['target'] = aggregate_df['filepath'].apply(
        lambda x: 0 if 'ses-placebo' in x else 1
    )
    # make sure each target has the same number of 0s and 1s, logg the distribution and 
    # drop subjects that have missing targets (i.e only 0s or only 1s)
    target_counts = aggregate_df.groupby('dataset')['target'].value_counts().unstack(fill_value=0)
    logging.info("Target distribution per dataset:")
    logging.info(target_counts)
    # Check if any subject has only one class and drop that subject from the aggregate dataframe
    single_class_subjects = aggregate_df.groupby('subject')['target'].nunique()
    single_class_subjects = single_class_subjects[single_class_subjects == 1].index
    if not single_class_subjects.empty:
        logging.warning(f"Found subjects with only one class in target: {single_class_subjects.tolist()}. "
                        f"Dropping these subjects from the aggregate dataframe.")
        aggregate_df = aggregate_df[~aggregate_df['subject'].isin(single_class_subjects)]
        # log again the target distribution after dropping single class subjects
        target_counts = aggregate_df.groupby('dataset')['target'].value_counts().unstack(fill_value=0)
        logging.info("Target distribution per dataset after dropping single class subjects:")
        logging.info(target_counts)

    # For the LSD dataset, change the dataset into dataset-task using the task column
    aggregate_df.loc[aggregate_df['dataset'] == 'lsd', 'dataset'] = aggregate_df['dataset'] + '-' + aggregate_df['task']
    logging.info("Splitted LSD dataset into individual task datasets.")
    logging.info(f"Unique datasets after splitting LSD: {aggregate_df['dataset'].unique()}")

    # Add an lsd-avg dataset that contains the average of all lsd tasks
    lsd_avg_df = aggregate_df[aggregate_df['dataset'].str.startswith('lsd-')].groupby(
        ['subject', 'target']
    ).mean(numeric_only=True).reset_index()
    lsd_avg_df['dataset'] = 'lsd-avg'
    lsd_avg_df = lsd_avg_df.reindex(columns=aggregate_df.columns)  # Ensure the same columns as aggregate_df
    aggregate_df = pd.concat([aggregate_df, lsd_avg_df], ignore_index=True)

    logging.info("Added lsd-avg dataset with average values of all LSD tasks.")
    logging.info(f"Total datasets after adding lsd-avg: {aggregate_df['dataset'].unique()}")

    # Drop unnecessary columns
    columns_to_drop = ['filepath', 'id', 'task']
    aggregate_df.drop(columns=columns_to_drop, inplace=True)
    logging.info(f"Dropped unnecessary columns: {columns_to_drop}.")
    # Ensure the subject column is the first column
    aggregate_df = aggregate_df[['subject'] + [col for col in aggregate_df.columns if col != 'subject']]
    logging.info("Reordered columns to have 'subject' as the first column.")

    # log the final shape of the aggregate dataframe
    logging.info(f"Final shape of the aggregate dataframe: {aggregate_df.shape}")
    # Save the cleaned aggregate dataframe
    cleaned_aggregate_file = "./local_data/aggregate_cleaned.csv"
    aggregate_df.to_csv(cleaned_aggregate_file, index=False)
    logging.info(f"Saved cleaned aggregate dataframe to {cleaned_aggregate_file}")
    # Split the aggregate dataframe into multiple csv files, one for each dataset
    for dataset in aggregate_df['dataset'].unique():
        dataset_df = aggregate_df[aggregate_df['dataset'] == dataset]
        # Drop the dataset column
        dataset_df = dataset_df.drop(columns=['dataset'])
        # log the shape of the dataset
        logging.info(f"Dataset {dataset} has shape: {dataset_df.shape}")
        dataset_file = f"./local_data/{dataset}.csv"
        dataset_df.to_csv(dataset_file, index=False)
        logging.info(f"Saved dataset {dataset} to {dataset_file}")
    logging.info("Finished splitting aggregate CSV into individual dataset CSV files.")
    logging.info("All datasets have been processed and saved successfully.")
    logging.info("Script completed successfully.")

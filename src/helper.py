import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def load_csv_files(data_dir):
    data = {}
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            name = file.replace(".csv", "")
            data[name] = pd.read_csv(os.path.join(data_dir, file)).convert_dtypes()


    return data

def convert_columns_to_datetime(convert, rename=False, unix_time=False):
    for df, col in convert:
        col_to_use = col
        if rename:
            col_to_use = ''.join(['_' + c.lower() if c.isupper() else c for c in col]).lstrip('_')
            df.rename(columns={col: col_to_use}, inplace=True)

        # Case 1: original data → parse as Unix
        if unix_time:
            df[col_to_use] = pd.to_datetime(df[col_to_use], unit='s', errors='coerce')

        # Case 2: cleaned data → parse as string-formatted datetime
        else:
            df[col_to_use] = pd.to_datetime(df[col_to_use], errors='coerce')

    print("===Converted timestamps===")


def prepare_data(path: str):
    data = load_csv_files(path)

    # build conversion list
    convert = []
    if "activity" in data:
        convert += [
            (data["activity"], 'activity_started'),
            (data["activity"], 'activity_completed'),
            (data["activity"], 'activity_updated'),
        ]
    if "math_results" in data:
        convert.append((data["math_results"], 'time'))
    if "essay_results" in data:
        convert.append((data["essay_results"], 'time'))
    if "text_results" in data:
        convert.append((data["text_results"], 'time'))
    if "all_scores" in data:
        convert.append((data["all_scores"], 'time'))

    unix_timestamp = path.endswith("/original")
    convert_columns_to_datetime(convert, rename=True, unix_time=unix_timestamp)

    return data


def visualize_feature():
    pass
    #TODO implement

def describe_feature():
    pass
    #TODO implement


def plot_histogram(data,
                   column_name=None,
                   title="Histogram",
                   xlabel=None,
                   ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if column_name is not None:
        data = data[column_name]

    sns.histplot(data, ax=ax, color="#D3D3D3", edgecolor='k')

    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel is not None else (column_name if column_name is not None else ""))

    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color="#66ff00", linestyle='dashed', linewidth=2, label=f'Mean: {round(mean_val, 2)}')
    ax.axvline(median_val, color="#ff33cc", linestyle='dashed', linewidth=2, label=f'Median: {round(median_val, 2)}')
    ax.legend()

    return ax
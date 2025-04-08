import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.helper import *

DATA_DIR = "data/original"


def add_domains(data):
    math_ids = set([str(m_id) for m_id in data["math_results"]["course_id"].unique()])
    text_ids = set([str(t_id) for t_id in data["text_results"]["course_id"].unique()])
    essay_ids = set([str(e_id) for e_id in data["essay_results"]["course"].unique()])
    activity_course_ids = set([str(c_id) for c_id in data["activity"]["course_id"].unique()])

    print(f"Overlap between math_ids and text_ids: {len(math_ids.intersection(text_ids))}")
    math_essay_id_overlap = math_ids.intersection(essay_ids)
    print(f"Overlap between math_ids and essay_ids: {len(math_essay_id_overlap)}")
    print(f"Overlap between text_ids and essay_ids: {len(text_ids.intersection(essay_ids))}")

    # print(f"The course ids {' & '.join(str(i) for i in math_ids.intersection(essay_ids))} exists both for math and essay courses."
    #       f"This complicates our task of mapping each course to a domain.")
    #
    # essay_types = list(data["essay_results"]["textType"].unique())
    # print(f"We notice that in all_scores, some test_ids clearly come from essay_results: {', '.join(essay_types)}.")
    # same_length = len(data["essay_results"]) == len(
    #     data["all_scores"][data["all_scores"]["test_id"].isin(essay_types)])
    #
    # if same_length:
    #     print(
    #         f"A quick check confirms that the number of entries in all_scores with these test_ids matches exactly the number of rows in essay_results "
    #         f"(Both are of length {len(data['essay_results'])}).")
    #     print("We can therefore use test_id to determine whether a result belongs to an essay or math test.")
    #     print("However, this only works for all_results.csv. For activity.csv, the problem of mapping remains.")
    #

    print("==Assigning domains==")
    def assign_domain(row):
        course_id = str(row["course"]) if "course" in row else str(row["course_id"])
        if course_id in text_ids:
            return "text"
        elif course_id in math_ids:
            return "math"
        elif course_id in essay_ids:
            return "essay"
        else:
            return pd.NA

    data["all_scores"]["domain"] = data["all_scores"].apply(assign_domain, axis=1)
    missing_vals_scores = data["all_scores"]["domain"].isna().sum()
    print(f"For all scores, all entries could be mapped to a domain. (missing_vals_scores:  {missing_vals_scores})")


    data["activity"]["domain"] = data["activity"].apply(assign_domain, axis=1)
    non_mappable_course_ids = data["activity"][data["activity"]["domain"].isna()]["course_id"].unique()
    missing_vals = data["activity"]["domain"].isna().sum()
    print(f"For {missing_vals} activities we could not assign a domain since their IDs don't match any of the provided ones. "
          f"({round(missing_vals/len(data['activity']), 2)*100}%)")
    print(f"Non mappable course_ids: {', '.join([str(x) for x in non_mappable_course_ids])}")


def inspect_missing_data(df: pd.DataFrame, df_name: str):
    print(f"==Inspecting missing data for {df_name}==")
    print(df.isnull().sum().reset_index(name='Nb of NAN'))

def visualize_dates(df: pd.DataFrame, column_name: str, plot=False):
    """
    Plots a histogram showing the distribution of dates in a given column of a DataFrame.
    """
    date_series = pd.to_datetime(df[column_name], errors='coerce').dropna()

    if plot:
        plt.figure(figsize=(10, 4))
        plt.hist(date_series, color='steelblue', edgecolor='black')
        plt.title(f'Distribution of Dates in {column_name}')
        plt.xlabel('Date')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

def restore_completed_before_started(activity: pd.DataFrame) -> pd.DataFrame:
    """
    Restores rows where activity_completed < activity_started by replacing
    activity_completed with activity_updated, but only if:
    - activity_updated is not null
    - activity_updated >= activity_started

    Marks restored rows in 'date_restored' column.
    Sets 'times_valid' to False for unfixable rows.
    """
    cond_completed_before_started = (
        activity['activity_completed'].notnull() &
        (activity['activity_completed'] < activity['activity_started'])
    )

    candidates = activity[cond_completed_before_started]

    updated_before_started = candidates[candidates["activity_updated"] < candidates["activity_started"]]
    updated_after_started = candidates[candidates["activity_updated"] > candidates["activity_started"]]
    updated_equals_started = candidates[candidates["activity_updated"] == candidates["activity_started"]]

    print(
        f"We can restore the rows where the activity was marked as completed before it started using the information provided by activity_updated. "
        f"\nSpecifically, we can use entries where 'activity_updated' was equal to or after 'activity_started' "
        f"({len(updated_equals_started)} + {len(updated_after_started)} rows).\n"
        f"However, in the {len(updated_before_started)} cases where 'activity_updated' is also before 'activity_started', we cannot determine the ground truth.\n"
        f"Since these cases represent only {round((len(updated_before_started) / len(activity)) * 100, 2)}% of the total data, we consider it acceptable to mark them as invalid."
    )

    # Restore fixed rows
    to_restore = updated_after_started.index.union(updated_equals_started.index)
    activity.loc[to_restore, "activity_completed"] = activity.loc[to_restore, "activity_updated"]
    activity.loc[:, "date_restored"] = activity.get("date_restored", False)
    activity.loc[to_restore, "date_restored"] = True

    # Set times_valid flags
    activity.loc[:, "times_valid"] = activity.get("times_valid", True)
    activity.loc[updated_before_started.index, "times_valid"] = False

    return activity


def restore_updated_before_started(activity: pd.DataFrame) -> pd.DataFrame:
    """
    Restores rows where activity_updated < activity_started by replacing
    activity_updated with activity_completed, but only if:
    - activity_completed is not null
    - activity_completed >= activity_started

    Adds a 'date_restored' column for fixed rows.
    Sets 'times_valid' to False for unfixable rows.
    """
    cond_updated_before_started = activity["activity_updated"] < activity["activity_started"]

    candidates = activity[cond_updated_before_started]
    can_restore = (
        candidates["activity_completed"].notna() &
        (candidates["activity_completed"] >= candidates["activity_started"])
    )
    cannot_restore = ~can_restore

    to_restore = candidates[can_restore].index
    to_invalidate = candidates[cannot_restore].index

    # Apply restoration
    activity.loc[to_restore, "activity_updated"] = activity.loc[to_restore, "activity_completed"]
    activity.loc[:, "date_restored"] = activity.get("date_restored", False)
    activity.loc[to_restore, "date_restored"] = True

    # Apply times_valid flag
    activity.loc[:, "times_valid"] = activity.get("times_valid", True)
    activity.loc[to_invalidate, "times_valid"] = False

    print(f"Restored {len(to_restore)} rows where 'activity_updated' was before 'activity_started' "
          f"by replacing it with 'activity_completed'.")
    print(f"Marked {len(to_invalidate)} rows as invalid due to unresolved inconsistency.")

    return activity


def clean_activity_dates(activity: pd.DataFrame):
    print(f"==Cleaning dates in activity.csv==")
    print("First, we will try to restore all values where: "
          "activity_updated < activity_started, "
          "or activity_completed < activity_started")

    cond_updated_before_started = activity['activity_updated'] < activity['activity_started']
    cond_completed_before_started = (
        activity['activity_completed'].notnull() &
        (activity['activity_completed'] < activity['activity_started'])
    )

    print(f"- Rows where activity_updated < activity_started: {cond_updated_before_started.sum()}")
    print(f"- Rows where activity_completed < activity_started: {cond_completed_before_started.sum()}")

    cond = cond_updated_before_started | cond_completed_before_started
    invalid_time_rows = activity[cond]
    invalid_time_rows_per = round(len(invalid_time_rows)/len(activity)*100, 2)
    print(f"In total, {len(invalid_time_rows)} rows have invalid times ({invalid_time_rows_per}% of the total data).")

    # === Apply fixes ===
    activity = restore_completed_before_started(activity)
    activity = restore_updated_before_started(activity)

    # Fill missing restoration flags
    activity['date_restored'] = activity['date_restored'].fillna(False)

    # Count final valid vs. invalid after restoration
    still_invalid = (
        (activity['activity_updated'] < activity['activity_started']) |
        (activity['activity_completed'].notnull() & (activity['activity_completed'] < activity['activity_started']))
    )

    activity['times_valid'] = ~still_invalid

    num_invalid_original = len(invalid_time_rows)
    num_invalid_remaining = still_invalid.sum()
    num_restored = num_invalid_original - num_invalid_remaining

    print(f"== Summary of cleaning process ==")
    print(f"Originally, {num_invalid_original} rows had inconsistent timestamps.")
    print(f"We were able to restore {num_restored} of these rows.")
    print(f"{num_invalid_remaining} rows still contain unresolved date inconsistencies.")

    if num_invalid_remaining > 0:
        print("Breakdown of unresolved rows (by user_id, course_id):")
        print(activity[still_invalid][["user_id", "course_id"]].value_counts().head())

    print(f"Since unresolved rows account for only {round((num_invalid_remaining / len(activity)) * 100, 2)}% of the total data, "
          f"we consider it acceptable to exclude them from time-related processing.")
    print(f"==Finished cleaning dates in activity.csv==")


def compare_times_from_activity_and_scores(activity: pd.DataFrame, all_scores: pd.DataFrame, plot=False):
    # only work on valid dates from activity.csv
    activity_valid = activity[activity['times_valid']]

    # Flatten all relevant datetime columns from activity
    activity_dates = pd.concat([
        activity_valid["activity_started"],
        activity_valid["activity_updated"],
        activity_valid["activity_completed"]
    ]).dropna()

    # Convert to timestamps for consistency
    activity_dates = pd.to_datetime(activity_dates)
    scores_dates = pd.to_datetime(all_scores["time"].dropna())

    # Create summary DataFrame
    summary_df = pd.DataFrame({
        "activity_dates": [activity_dates.min(), activity_dates.max(), activity_dates.count()],
        "scores_dates": [scores_dates.min(), scores_dates.max(), scores_dates.count()]
    }, index=["min", "max", "count"])

    # Plot histograms
    if plot:
        plt.figure(figsize=(10, 5))

        min_date = min(activity_dates.min(), scores_dates.min())
        max_date = max(activity_dates.max(), scores_dates.max())
        bins = pd.date_range(start=min_date, end=max_date, freq='W')

        plt.hist(activity_dates, bins=bins, color='blue', alpha=0.6, label='Activity Dates')
        plt.hist(scores_dates, bins=bins, color='yellow', alpha=1, label='Scores Dates')

        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.title("Weekly Distribution of Activity and Score Dates")
        plt.tight_layout()
        plt.show()

    print(summary_df)
    dataset_date_range = (max(activity_dates.max(), scores_dates.max()) - min(activity_dates.min(), scores_dates.min())).days
    print(f"We can see that we have around a factor {round(activity_dates.count() / scores_dates.count(), 2)} more dates in activities than in scores. \n"
          f"In total, the dataset covers a range of {dataset_date_range} days. \n"
          f"We notice that for the last {(activity_dates.max() - scores_dates.max()).days} days in the dataset, we only have data from activity.csv, but not from scores.csv.")


def clean():
    # Load all available data
    data = prepare_data(DATA_DIR)

    activity = data["activity"]
    essay_results = data["essay_results"]
    math_results = data["math_results"]
    text_results = data["text_results"]
    all_scores = data["all_scores"]
    print("===Loaded all data===")

    # convert unix timestamps to datetime
    convert_columns_to_datetime([
        (activity, 'activity_started'),
        (activity, 'activity_completed'),
        (activity, 'activity_updated'),
        (math_results, 'time'),
        (essay_results, 'time'),
        (text_results, 'time'),
        (all_scores, 'time'),
    ], rename=True)
    print("===Converted timestamps===")

    # add domain to activity.csv
    print("Adding domain (math, essay, text) to all_scores.csv and activity.csv")
    add_domains(data)

    print("===Visualize dates===")
    visualize_dates(activity, 'activity_started')
    visualize_dates(activity, 'activity_updated')
    visualize_dates(activity, 'activity_completed')
    print("It becomes clear that there are some wrong dates in activity_completed in activity.csv, \n"
          "as a couple of dates are from 1970, and sometimes the order of activities is non-sensical (e.g. activities finishing before they started). \n"
          "We will handle these values.")
    visualize_dates(all_scores, 'time')

    clean_activity_dates(activity)
    compare_times_from_activity_and_scores(activity, all_scores, plot=True)
    print(f"The scores times lie in a reasonable range (between {all_scores.date.min()} and {all_scores.date.max()}).\n)"
          f"There is also only one time column ('date'), so there are no conflicts with order of events."
          f"We therefore don't need to clean them.")

    # Inspect missing data
    inspect_missing_data(activity, 'activity')
    print("In the activity dataset, the activity_completed and domain columns are the only ones with missing values. "
          "\nA missing value in activity_completed column indicates that the activity was not completed, either because "
          "\nit was abandoned or because completions aren’t recorded (as with access and exams in course IDs 3301 and 5447).")


    inspect_missing_data(all_scores, 'all_scores')
    print("all_scores is not missing any data.")

    activity.to_csv("data/cleaned/activity.csv", index=False)
    all_scores.to_csv("data/cleaned/all_scores.csv", index=False)
    print("===Finished data cleaning. Wrote the updated file to data/cleaned===")

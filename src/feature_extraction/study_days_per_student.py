import matplotlib.pyplot as plt
from src.helper import *

import pandas as pd
from typing import Dict

def compute_user_activity_days(activity: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'user_activity_day' column to the activity DataFrame.
    It indicates the N-th unique day the user was active.
    """
    activity = activity.copy()
    activity['activity_date'] = activity['activity_started'].dt.date

    # Drop duplicates to get one entry per user per day
    user_day_df = activity[['user_id', 'activity_date']].drop_duplicates()

    # Sort and assign activity day index per user
    user_day_df['user_activity_day'] = (
        user_day_df.sort_values(['user_id', 'activity_date'])
                   .groupby('user_id')
                   .cumcount()
                   .add(1)
    )

    # Merge the activity day index back to the original activity DataFrame
    activity = activity.merge(user_day_df, on=['user_id', 'activity_date'])
    return activity.drop(columns='activity_date')


def compute_user_exam_days(all_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'user_exam_day' column to the all_scores DataFrame.
    It indicates the N-th unique day the user took an exam.
    """
    all_scores = all_scores.copy()
    all_scores['exam_date'] = all_scores['time'].dt.date

    user_exam_df = all_scores[['user_id', 'exam_date']].drop_duplicates()
    user_exam_df['user_exam_day'] = (
        user_exam_df.sort_values(['user_id', 'exam_date'])
                    .groupby('user_id')
                    .cumcount()
                    .add(1)
    )

    all_scores = all_scores.merge(user_exam_df, on=['user_id', 'exam_date'])
    return all_scores.drop(columns='exam_date')


def compute_study_and_exam_days(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Adds 'user_activity_day' and 'user_exam_day' to activity and all_scores DataFrames.
    Returns a dictionary with updated DataFrames.
    """
    activity = compute_user_activity_days(data['activity'])
    all_scores = compute_user_exam_days(data['all_scores'])
    visualize_activity_and_exam_days(activity, all_scores)
    return {'activity': activity, 'all_scores': all_scores}


def visualize_activity_and_exam_days(activity: pd.DataFrame, all_scores: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    plot_histogram(
        data=activity,
        column_name="user_activity_day",
        title="Distribution of User Activity Days",
        xlabel="Activity Day",
        ax=axes[0]
    )

    plot_histogram(
        data=all_scores,
        column_name="user_exam_day",
        title="Distribution of User Exam Days",
        xlabel="Exam Day",
        ax=axes[1]
    )

    plt.tight_layout()
    plt.show()
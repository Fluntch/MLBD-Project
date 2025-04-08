import matplotlib.pyplot as plt
from src.helper import *
from src.const import *

import pandas as pd
from typing import Dict

def compute_user_days(activity: pd.DataFrame, all_scores: pd.DataFrame) -> pd.DataFrame:
    activity = activity.copy()
    all_scores = all_scores.copy()

    activity['type'] = 'activity'
    all_scores['type'] = 'exam'

    combined = pd.concat([
        activity[['user_id', 'date', 'type']],
        all_scores[['user_id', 'date', 'type']]
    ])

    combined = combined.drop_duplicates(['user_id', 'date', 'type'])

    interaction_type = (
        combined.groupby(['user_id', 'date'])['type']
        .agg(lambda x: 'both' if set(x) == {'activity', 'exam'} else x.iloc[0])
        .reset_index()
    )

    interaction_type['user_day'] = (
        interaction_type.sort_values(['user_id', 'date'])
        .groupby('user_id')
        .cumcount()
        .add(1)
    )

    return interaction_type

def compute_study_and_exam_days(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    user_days = compute_user_days(data['activity'], data['all_scores'])
    if PLOT:
        visualize_user_days(user_days)
    return user_days

def visualize_user_days(combined: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    plot_histogram(
        data=combined,
        column_name="user_day",
        title="All Interactions",
        xlabel="User Day",
        ax=axes[0]
    )

    plot_histogram(
        data=combined[combined["type"] == "activity"],
        column_name="user_day",
        title="Activity Only",
        xlabel="User Day",
        ax=axes[1]
    )

    plot_histogram(
        data=combined[combined["type"] == "exam"],
        column_name="user_day",
        title="Exam Only",
        xlabel="User Day",
        ax=axes[2]
    )

    plt.tight_layout()
    plt.show()
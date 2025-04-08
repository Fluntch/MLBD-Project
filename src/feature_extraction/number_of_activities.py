import pandas as pd
from src.helper import *
from src.const import *

def visualize_number_of_activities(user_days: pd.DataFrame):
    plot_histogram(user_days.number_of_activities, title = 'Number of Activities per Day')
    plt.show()


def compute_number_of_activities(user_days: pd.DataFrame, activity: pd.DataFrame):
    """ For each user x date combination, extract the number of activities """

    activity_counts = activity.groupby(["user_id", "date"]).size().reset_index(name="number_of_activities")
    user_days = user_days.merge(activity_counts, on=["user_id", "date"], how="left")
    user_days["number_of_activities"] = user_days["number_of_activities"].fillna(0).astype(int)

    if PLOT:
        visualize_number_of_activities(user_days)

    return user_days





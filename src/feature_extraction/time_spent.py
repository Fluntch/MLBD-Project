import pandas as pd
from matplotlib import pyplot as plt


def truncate_long_durations(activity: pd.DataFrame, activity_type: str, perc: float = 0.75):
    """Truncate very long durations per activity_type based on a percentile threshold."""
    threshold = activity[activity.activity_type == activity_type]["time_in_minutes"].quantile(perc)

    mask = activity["activity_type"] == activity_type
    long_mask = mask & (activity["time_in_minutes"] > threshold)

    if "time_truncated" not in activity.columns:
        activity["time_truncated"] = False
    activity.loc[long_mask, "time_truncated"] = True

    activity.loc[long_mask, "time_in_minutes"] = threshold

    print(f"{long_mask.sum()} entries were truncated for activity_type '{activity_type}' (>{perc*100:.0f}th percentile = {threshold:.2f} min)")
    title = "Time spent in " + activity_type + " activity (after truncation)"
    visualize_time_spent(activity, activity_type, title=title)

    return activity


def visualize_time_spent(activity: pd.DataFrame, activity_type: str, title: str = None):
    data = activity[activity.activity_type == activity_type]["time_in_minutes"]
    percentile_75 = data.quantile(0.75)

    plt.hist(data, bins=30, edgecolor='black')
    plt.axvline(percentile_75, color='red', linestyle='--', linewidth=2, label='75th percentile')
    plt.xlabel("Time in minutes")
    plt.ylabel(f"{activity_type} count")

    if title is not None:
        plt.title(title)
    else:
        plt.title(f"Time spent in {activity_type} activity")

    plt.legend()
    plt.show()

def get_time_spent_overview(activity: pd.DataFrame):

    valid_times = pd.to_timedelta(activity["time_spent"].dropna())
    activity["time_in_minutes"] = valid_times.dt.total_seconds() / 60

    gb = activity.groupby(['activity_type', 'domain']).describe()['time_in_minutes']
    gb["z_score_max"] = (gb["max"] - gb["max"].mean())/gb["max"].std()
    print(gb)
    print(f"There are five kind of activities: access, course, exam, lesson and topic.")
    print(f"We will focus on the activity types course, exam, lesson and topic.\n"
          f" We will exclude access since 75% values for all three domains lie between 0.0 and 0.1. "
          f"while the mean is much higher (between 1.71 and 2.25) - indicating very few, very strong outliers. "
          f"These outliers could lead to misleading results in the analysis.")
    activities = ["course", "exam", "lesson", "topic"]
    for activity_type in activities:
        visualize_time_spent(activity, activity_type)
    print("Let's exclude topics and lesson due to their very strong variation until we get an answer on what a realistic range is.")

    print("We will truncate times using the .75 percentile") #TODO we can't just use a one solution fits all approach here. We need to individually inspect the data
    for activity_type in activities:
        activity = truncate_long_durations(activity, activity_type)


def compute_time_spent(activity: pd.DataFrame):
    """ For each activity recorded in activity.csv, compute how much time was spent on the activity
    If an activity was not finished, use last updated time as measurement """
    def add_time_spent(row):
        if row.times_valid:
            if not pd.isnull(row.activity_completed):
                return row.activity_completed - row.activity_started
            else:
                return row.activity_updated - row.activity_started
        else:
            return pd.NA


    print("==Computing time spent==")
    activity["time_spent"] = activity.apply(add_time_spent, axis=1)
    comp_worked_per = round((activity["time_spent"].notna().sum() / len(activity)) * 100, 2)
    print(f"Time spent could be computed for {comp_worked_per}% of activities")

    get_time_spent_overview(activity)

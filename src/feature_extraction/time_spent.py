from src.config import PLOT
from src.helper import *

def truncate_long_durations(activity: pd.DataFrame):
    """Truncate very long durations per activity_type based on a fixed duration threshold."""

    max_duration = {
        "exam": 60 + 15,
        "lesson": 60 * 24 * 5,
        "topic": 30,
        "course": 60 * 24 * 30 * 12
    }

    explanation = (
        "The truncation thresholds are chosen based on domain-specific guidance:\n"
        "- Exams are designed to simulate real exams and typically take 45–60 minutes. We allow up to 75 minutes.\n"
        "- Lessons are intended to span up to a week of real-time engagement, with approximately 5 platform hours. We cap at 5 days.\n"
        "- Topics take about 7 minutes to read, but can go longer if students engage deeply. We cap at 30 minutes.\n"
        "- Courses run in parallel and are typically used over 7 months to a year. We allow up to 12 months."
    )
    print(explanation)

    for activity_type, threshold in max_duration.items():
        mask = activity.activity_type == activity_type
        long_mask = mask & (activity.time_in_minutes >= threshold)

        num_truncated = long_mask.sum()
        total = mask.sum()
        truncated_perc = round((num_truncated / total) * 100, 2) if total > 0 else 0

        activity.loc[long_mask, "time_spent"] = pd.to_timedelta(threshold, unit="m")
        activity.loc[long_mask, "time_in_minutes"] = threshold
        activity.loc[long_mask, "time_truncated"] = True

        print(f"{num_truncated} entries were truncated for activity_type '{activity_type}' "
              f"(>= {threshold} min, {truncated_perc}% of {total} entries)")

        title = f"Time spent in {activity_type} activity (after truncation)"
        if PLOT:
            visualize_time_spent(activity, activity_type, title=title)

    activity["time_truncated"] = activity["time_truncated"].fillna(False)
    return activity


def visualize_time_spent(activity: pd.DataFrame, activity_type: str, title: str = None):
    data = activity[activity.activity_type == activity_type]["time_in_minutes"]
    percentile_75 = data.quantile(0.75)

    plt.hist(data, bins=30, edgecolor='black')
    plt.axvline(percentile_75, color='red', linestyle='--', linewidth=2, label='75th percentile')
    plt.xlabel("Time in minutes")
    plt.ylabel(f"{activity_type} count")

    title = title or f"Time spent in {activity_type} activity"
    plt.title(title)

    plt.legend()
    save_plot(plt.gcf(), __file__, title)
    plt.close()

def get_time_spent_overview(activity: pd.DataFrame) -> pd.DataFrame:
    activity = activity.copy()

    # First, remove unrealistic durations before any conversion
    max_duration = pd.Timedelta(hours=5, minutes=0)
    before_len = len(activity)
    activity = activity[activity["time_spent"] <= max_duration]
    removed = before_len - len(activity)
    if removed > 0:
        removed_perc = round((removed / before_len) * 100, 2)
        print(f"Removed {removed} entries with time_spent > 5 hours ({removed_perc}%)")

    # Then convert to minutes
    valid_times = pd.to_timedelta(activity["time_spent"].dropna())
    activity["time_in_minutes"] = valid_times.dt.total_seconds() / 60

    gb = activity.groupby(['activity_type', 'domain']).describe()['time_in_minutes']
    gb["z_score_max"] = (gb["max"] - gb["max"].mean()) / gb["max"].std()
    print(gb)

    print(f"There are five kind of activities: access, course, exam, lesson and topic.")
    print(f"We will focus on the activity types course, exam, lesson and topic.\n"
          f" We will exclude access since 75% values for all three domains lie between 0.0 and 0.1. "
          f"while the mean is much higher (between 1.71 and 2.25) - indicating very few, very strong outliers. "
          f"These outliers could lead to misleading results in the analysis.")

    activities = ["course", "exam", "lesson", "topic"]
    if PLOT:
        for activity_type in activities:
            visualize_time_spent(activity, activity_type)

    print("Let's exclude topics and lesson due to their very strong variation until we get an answer on what a realistic range is.")
    print("We will truncate times using the .75 percentile")

    activity = truncate_long_durations(activity)
    return activity

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

    return get_time_spent_overview(activity)

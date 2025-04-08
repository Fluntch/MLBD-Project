from src.helper import *
from src.config import PLOT

def visualize_time_spent_per_day(user_data, activity):
    print("Plotting, this may take a while...")
    activities = [a for a in activity["activity_type"].unique() if a != "access"]
    domains = ["math", "essay", "text"]

    fig, axes = plt.subplots(len(activities), len(domains), figsize=(6 * len(domains), 4 * len(activities)))

    for i, a in enumerate(activities):
        for j, domain in enumerate(domains):
            ax = axes[i][j] if len(activities) > 1 else axes[j]
            subset = user_data[(user_data["activity_type"] == a)
                               & (user_data["domain"] == domain)]["time_in_minutes"]

            plot_histogram(
                data=subset,
                title=f"{a.title()} • {domain.title()}",
                xlabel="Minutes",
                ax=ax
            )

    plt.tight_layout()
    save_plot(plt.gcf(), __file__, "Time Spent per Day")
    plt.close()

def compute_time_spent_per_day(user_days: pd.DataFrame, activity: pd.DataFrame):
    """per activity, total, ..."""
    time_spent_per_day = activity.groupby(["user_id", "date", "domain", "activity_type"])["time_in_minutes"].sum().reset_index()

    #NOTE: these are almost the same graphs as computed in time_spent, however, this time their are computed by day and domain.
    if PLOT:
        visualize_time_spent_per_day(time_spent_per_day, activity)

    user_days = user_days.merge(time_spent_per_day, on=["user_id", "date"], how="left")
    return user_days

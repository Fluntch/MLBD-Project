from src.feature_extraction.test_difficulty import get_test_performance

def compute_average_performance(all_scores, user_days):
    percentages = all_scores.groupby(["user_id", "test_id", "course", "date"])["percentage"].mean().reset_index()
    percentages["performance"] = percentages.apply(
        lambda row: get_test_performance(
            test_id=row["test_id"],
            course=row["course"],
            percentage=row["percentage"]
        ),
        axis=1
    )

    user_days = user_days.merge(percentages, on=["user_id", "date"], how="left")

    return user_days


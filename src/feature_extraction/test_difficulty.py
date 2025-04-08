import pandas as pd
import matplotlib.pyplot as plt

AVERAGE_SCORES_PATH = "data/aggregated/average_test_scores.csv"

def compute_average_test_scores(all_scores: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Computes and optionally saves and plots the average test scores per test_id and course.
    """
    average_test_scores = (
        all_scores.groupby(['test_id', 'course'])
        .mean(numeric_only=True)['percentage']
        .apply(lambda x: round(x, 2))
        .reset_index()
    )

    print(average_test_scores['percentage'].describe())

    if save:
        average_test_scores.to_csv(AVERAGE_SCORES_PATH, index=False)

    return average_test_scores


def get_test_performance(test_id: str, course: str, percentage: float) -> float:
    """
    Returns the deviation of the given percentage from the average score for a test.
    A positive deviation indicates above-average performance.
    """
    average_test_scores = pd.read_csv(AVERAGE_SCORES_PATH)

    match = average_test_scores[
        (average_test_scores['test_id'] == str(test_id)) &
        (average_test_scores['course'] == course)
    ]

    if match.empty:
        raise ValueError(f"No average score found for test_id='{test_id}', course='{course}'")

    avg_score = match['percentage'].iloc[0]
    return percentage - avg_score

def visualize_test_performance(activity: pd.DataFrame):
    plt.hist(activity['performance'], edgecolor="black")
    plt.title("Distribution of Test Performances")
    plt.xlabel("Relative Test Performance (in %)")
    plt.ylabel("Count")

    plt.show()

def compute_test_difficulty(all_scores: pd.DataFrame):
    """
    Returns tests sorted by difficulty (based on average scores).
    Lower scores indicate higher difficulty.
    """
    print("===Test difficulty===")
    average_scores = compute_average_test_scores(all_scores)
    sorted_scores = average_scores.sort_values(by='percentage').reset_index(drop=True)

    print("Top 5 most difficult tests (lowest average percentage):")
    print(sorted_scores.head())

    print("==Adding performances to all_scores==")
    all_scores["performance"] = all_scores.apply(
        lambda row: get_test_performance(
            test_id=row["test_id"],
            course=row["course"],
            percentage=row["percentage"]
        ),
        axis=1
    )

    visualize_test_performance(all_scores)


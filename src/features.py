from src.helper import *
from .feature_extraction import *

def extract():
    path = "data/cleaned"
    data = prepare_data(path)
    activity = data["activity"]
    all_scores = data["all_scores"]

    # time spent
    activity = compute_time_spent(activity)

    # test difficulty
    compute_test_difficulty(all_scores)

    # study days per student
    user_days = compute_study_and_exam_days(data)

    # number_of_activities_per_day
    user_days = compute_number_of_activities(user_days, activity)

    # time_spent_per_day
    user_days = compute_time_spent_per_day(user_days, activity)

    # avg_performance
    performances = compute_average_performance(all_scores)

    # avg_number_of_questions_attempted

    # avg_number_of_questions_solved

    # avg_time_between_consecutive_exams

    # usage_frequency(worked on the platform for at least 15 minutes)

    user_days.to_csv("data/features/user_days.csv", index=False)
    performances.to_csv("data/features/performances.csv", index=False)
    activity.to_csv("data/features/activity.csv", index=False)

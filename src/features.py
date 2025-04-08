from src.helper import *
from .feature_extraction import *



def extract():
    path = "data/cleaned"
    data = prepare_data(path)
    activity = data["activity"]
    all_scores = data["all_scores"]


    # time spent
    # compute_time_spent(activity)

    # test difficulty
    # compute_test_difficulty(all_scores)

    # study days per student
    user_days = compute_study_and_exam_days(data)

    # avg_number_of_activities_per_day
    compute_number_of_activities(user_days, activity)

    # compute_number_of_activities(data["activity"])

    # avg_time_spent_per_session_per_day(activity_completed - activity_started)

    # avg_number_of_tests_taken

    # avg_number_of_questions_attempted

    # avg_number_of_questions_solved

    # avg_number_of_reattempts

    # avg_time_between_consecutive_tests

    # usage_frequency(worked on the plattform for at least 15 minutes)

    # activity_diversity

    pass
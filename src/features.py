from src.helper import *
from .feature_extraction import *



def extract():
    path = "data/cleaned"
    data = prepare_data(path)

    # time spent
    # compute_time_spent(data["activity"])

    # test difficulty
    # compute_test_difficulty(data["all_scores"])

    # study days per student
    compute_study_and_exam_days(data)

    # avg_number_of_activities

    # avg_time_spent_per_session(activity_completed - activity_started)

    # avg_number_of_tests_taken

    # avg_number_of_questions_attempted

    # avg_number_of_questions_solved

    # avg_number_of_reattempts

    # avg_time_between_consecutive_tests

    # usage_frequency(worked on the plattform for at least 15 minutes)

    # activity_diversity

    pass
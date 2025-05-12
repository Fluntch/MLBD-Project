# merge with mapped
# filter out chats < 2 messages (message_count)
# filter out non-math interactions
# create column in features: chatbot interactions per day (in #messages)
import pandas as pd


def compute_chatbot_interactions(user_data):
    mapping = pd.read_csv("data/original/mapping.csv", index_col=0)
    gymitrainer = pd.read_csv("data/original/gymitrainer.csv", index_col=0)
    gymitrainer["startTime"] = pd.to_datetime(gymitrainer["startTime"], unit="s")
    gymitrainer["date"] = gymitrainer["startTime"].dt.date
    gymitrainer = gymitrainer[gymitrainer["message_count"] > 1]
    gymitrainer = mapping.merge(gymitrainer, on="id", how="right")
    gymitrainer = gymitrainer[["user_id", "date", "message_count"]]

    user_data.date = user_data.date.astype("datetime64[ns]")
    gymitrainer.date = gymitrainer.date.astype("datetime64[ns]")

    user_data = user_data.merge(gymitrainer, on=["user_id", "date"], how="outer")

    return user_data

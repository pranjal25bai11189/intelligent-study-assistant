import pickle
import pandas as pd
from rules import give_advice


# Load model
model, le_time, le_mood = pickle.load(open("model.pkl", "rb"))

# Take input from user
sleep = float(input("Enter sleep hours: "))
study = float(input("Enter study hours: "))
time = input("Enter time of day (morning/afternoon/evening/night): ")
mood = input("Enter mood (fresh/normal/tired): ")

# Encode input
time_encoded = le_time.transform([time])[0]
mood_encoded = le_mood.transform([mood])[0]

# Predict
input_data = pd.DataFrame([[sleep, study, time_encoded, mood_encoded]],
                          columns=['sleep_hours', 'study_hours', 'time_of_day', 'mood'])
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][prediction]

# Output
print("\n--- RESULT ---")
print("Focus Prediction:", "High" if prediction == 1 else "Low")
print("Confidence:", round(prob * 100, 2), "%")
advice = give_advice(prediction, sleep, time)
print("Advice:", advice)
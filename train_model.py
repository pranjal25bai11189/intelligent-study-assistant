import pandas as pd
import pandas
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("data.csv", sep="\t")
df.columns = df.columns.str.strip()
print(df.head())

# Encode text into numbers
le_time = LabelEncoder()
le_mood = LabelEncoder()

df['time_of_day'] = le_time.fit_transform(df['time_of_day'])
df['mood'] = le_mood.fit_transform(df['mood'])

# Features and target
X = df[['sleep_hours', 'study_hours', 'time_of_day', 'mood']]
y = df['focus']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
pickle.dump((model, le_time, le_mood), open("model.pkl", "wb"))

print("Model trained successfully")
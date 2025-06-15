from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'classes_attended': [1, 0, 1, 0, 1, 1, 0, 1],
    'pass_exam': [0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['hours_studied', 'classes_attended']]
Y = df['pass_exam']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


model = RandomForestClassifier()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}")
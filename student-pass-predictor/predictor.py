import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("data.csv")

X = data[["Hours", "SleepHours"]]
Y = data["Passed"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)

print(f"Model accuracy: {accuracy * 100:.2f}%")

hours = float(input("Enter number of hours studied: "))
sleep_hours = float(input("Enter number of hours slept: "))

prediction = model.predict([[hours, sleep_hours]])
if prediction[0] == 1:
    print("✅ Student will likely PASS the exam.")
else:
    print("❌ Student will likely FAIL the exam.")
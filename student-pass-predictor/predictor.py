import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("data.csv")

X = data[["Hours"]]
Y = data["Passed"]

model = LogisticRegression()
model.fit(X, Y)

hours = float(input("Enter number of hours studied: "))

prediction = model.predict([[hours]])
if prediction[0] == 1:
    print("✅ Student will likely PASS the exam.")
else:
    print("❌ Student will likely FAIL the exam.")
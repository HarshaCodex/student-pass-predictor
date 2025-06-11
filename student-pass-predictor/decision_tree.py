import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'classes_attended': [1, 1, 2, 3, 3, 4, 5, 5],
    'pass_exam': [0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["hours_studied", "classes_attended"]]

Y = df["pass_exam"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(12,8))
plot_tree(model, feature_names=['hours_studied', 'classes_attended'], class_names=['Fail', 'Pass'], filled=True)
plt.show()




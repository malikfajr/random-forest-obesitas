import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def evaluate_model(model_name, true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print(f'{model_name} Model Evaluation:')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')


def plot_confusion_matrix(conf_matrix, classes, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()
     
data = pd.read_csv('obesitas.csv')
print(data.head())


missing_data = data.isnull().sum()
print(missing_data)

data.duplicated().sum()

data.drop_duplicates(inplace=True)


columns = data.columns
print(columns)


X = data.iloc[:, :-1]
y = data.iloc[:, -1]


X = pd.get_dummies(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.15, random_state=125)
# Random Forest Classifier
model = RandomForestClassifier(n_estimators=10, random_state=125)
model.fit(X_train, y_train_encoded)

# Evaluate model
predictions = model.predict(X_test)
evaluate_model("Decision Tree", y_test_encoded, predictions)



conf_mat_decision_tree = confusion_matrix(y_test_encoded, predictions)
plot_confusion_matrix(conf_mat_decision_tree, label_encoder.classes_, "Random Forest Confusion Matrix")

joblib.dump(model, 'model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
joblib.dump(X.columns, 'model_columns.joblib')
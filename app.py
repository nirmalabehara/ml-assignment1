import os

# Path to your model file
file_path = 'C:/Users/nirma/Downloads/archive (2)/best_random_forest_model.pkl'

# Check if the file exists
if os.path.isfile(file_path):
    print("File exists.")
else:
    print("File does not exist. Please check the path.")
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load your model
best_model = joblib.load('C:/Users/nirma/Downloads/archive (2)/best_random_forest_model.pkl')

# Define your feature names
feature_names = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 
                  'exng', 'oldpeak', 'slp', 'caa', 'thall']

# Create a DataFrame with your new data
new_data = pd.DataFrame([[98.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], columns=feature_names)

# Make predictions
predictions = best_model.predict(new_data)
print("Predictions:", predictions)
print("Class labels:", best_model.classes_)

# Load test data and labels
test_data = pd.read_csv('C:/Users/nirma/Downloads/archive (2)/heart.csv')
test_labels = pd.read_csv('C:/Users/nirma/Downloads/archive (2)/o2Saturation.csv')

# Prepare test features and labels
X_test = test_data.drop(columns=['output'])
y_test = test_data['output']

# Make predictions on test data
test_predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)
report = classification_report(y_test, test_predictions)

print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

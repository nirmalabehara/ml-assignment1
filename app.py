import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
best_model = joblib.load('C:/Users/nirma/Downloads/archive (2)/best_random_forest_model.pkl')
feature_names = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 
                  'exng', 'oldpeak', 'slp', 'caa', 'thall']
new_data = pd.DataFrame([[98.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], columns=feature_names)
predictions = best_model.predict(new_data)
print("Predictions:", predictions)
print("Class labels:", best_model.classes_)
test_data = pd.read_csv('C:/Users/nirma/Downloads/archive (2)/heart.csv')
test_labels = pd.read_csv('C:/Users/nirma/Downloads/archive (2)/o2Saturation.csv')
X_test = test_data.drop(columns=['output'])
y_test = test_data['output']
test_predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)
report = classification_report(y_test, test_predictions)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
joblib.dump(best_model, 'C:/Users/nirma/Downloads/archive (2)/updated_best_random_forest_model.pkl')

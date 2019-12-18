from sklearn.externals import joblib
from processing_data import test, convert_y_to_str


model = joblib.load('cls.pkl')
cv = joblib.load('CV.pkl')
prediction_data, y_result = test(cv, 'test_data.csv')
predicted_prob = model.predict_proba(prediction_data)
correct = True
for i, prob in enumerate(predicted_prob):
    if y_result[i] != convert_y_to_str(prob):
        correct = False
print(correct)

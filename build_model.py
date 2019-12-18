from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from processing_data import preprocess_data
from sklearn.externals import joblib


x, y = preprocess_data('training_data.csv')
linear_svc = LinearSVC()
calibrated_svc = CalibratedClassifierCV(linear_svc,
                                        method='sigmoid',
                                        cv=3)
calibrated_svc.fit(x, y)
joblib.dump(calibrated_svc, 'cls.model')

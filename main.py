from processing_data import query, convert_y_to_str
import numpy as np
from sklearn.externals import joblib


if __name__ == "__main__":
    while True:
        print('Input line (0 - exit):')
        line = str(input())
        if line == '0':
            break
        model = joblib.load('cls.pkl')
        cv = joblib.load('CV.pkl')
        test = query(cv, np.array([line]))
        print(convert_y_to_str(model.predict_proba(test)[0]))



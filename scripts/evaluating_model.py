import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

def evaluate_model(model_path, X_test_path, y_test_path):
    model = joblib.load(model_path)
    
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    
    X_test['year'] = pd.to_datetime(X_test['year']).dt.year
    
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    
    return mse, rmse

model_path = '../models/co2_emission_predictor_model.pkl'
X_test_path = '../data/X_test.csv'
y_test_path = '../data/y_test.csv'

mse, rmse = evaluate_model(model_path, X_test_path, y_test_path)

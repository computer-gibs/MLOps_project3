import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_model(X_train_path, y_train_path, model_save_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    
    # сохранение обученной модели
    joblib.dump(model, model_save_path)
    
    return model

X_train_path = '../data/X_train.csv'
y_train_path = '../data/y_train.csv'

# путь для сохранения обученной модели
model_save_path = '../models/co2_emission_predictor_model.pkl'

trained_model = train_model(X_train_path, y_train_path, model_save_path)
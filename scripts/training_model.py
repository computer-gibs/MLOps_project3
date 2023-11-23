import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model(X_train_path, y_train_path, model_save_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # преобразование категориальных переменных в числовые
    label_encoder = LabelEncoder()
    X_train['country_code'] = label_encoder.fit_transform(X_train['country_code'])

    # создание и обучение модели
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    
    joblib.dump(model, model_save_path)
    joblib.dump(label_encoder, '../models/label_encoder.pkl')
    
    return model

X_train_path = '../data/X_train.csv'
y_train_path = '../data/y_train.csv'

model_save_path = '../models/co2_emission_predictor_model.pkl'
trained_model = train_model(X_train_path, y_train_path, model_save_path)
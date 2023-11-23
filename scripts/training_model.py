import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def train_model(X_train_path, y_train_path, model_save_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    
    X_train['year'] = pd.to_datetime(X_train['year']).dt.year
    
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough')
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
    
    pipeline.fit(X_train, y_train.values.ravel())
    
    joblib.dump(pipeline, model_save_path)
    
    return pipeline

X_train_path = '../data/X_train.csv'
y_train_path = '../data/y_train.csv'

model_save_path = '../models/co2_emission_predictor_model.pkl'

trained_model = train_model(X_train_path, y_train_path, model_save_path)
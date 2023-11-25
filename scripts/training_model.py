import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import mlflow
from mlflow import log_metric, log_param, log_artifacts

mlflow.start_run()

current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '..', 'data')
models_dir = os.path.join(current_dir, '..', 'models')
train_file_path = os.path.join(data_dir, 'train_set.csv')
train_data = pd.read_csv(train_file_path)

# 'value' - это целевая переменная, а 'year' - единственная предикторная переменная
X = train_data[['year']]
y = train_data['value']

# создание и обучение модели линейной регрессии
model = LinearRegression()

log_param("model_type", "linear_regression")
log_param("features", X.columns.tolist())

model.fit(X, y)

log_metric("training_score", model.score(X, y))

model_file_path = os.path.join(models_dir, 'co2_emissions_model.joblib')
joblib.dump(model, model_file_path)
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()
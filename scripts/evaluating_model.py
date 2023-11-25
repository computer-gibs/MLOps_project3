import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import mlflow
from mlflow import log_metric

mlflow.start_run()

current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '..', 'data')
models_dir = os.path.join(current_dir, '..', 'models')

test_file_path = os.path.join(data_dir, 'test_set.csv')
test_data = pd.read_csv(test_file_path)

X_test = test_data[['year']]
y_test = test_data['value']

model_file_path = os.path.join(models_dir, 'co2_emissions_model.joblib')
model = joblib.load(model_file_path)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

log_metric("mae", mae)
log_metric("mse", mse)

print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
print(f"Среднеквадратическая ошибка (MSE): {mse:.2f}")

results_file_path = os.path.join(models_dir, 'model_evaluation_results.txt')
with open(results_file_path, 'w') as file:
    file.write(f"Средняя абсолютная ошибка (MAE): {mae:.2f}\n")
    file.write(f"Среднеквадратическая ошибка (MSE): {mse:.2f}\n")

log_artifacts(models_dir)

mlflow.end_run()
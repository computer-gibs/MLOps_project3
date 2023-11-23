import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(data_path):
    data = pd.read_csv(data_path)

    # что целевая переменная - 'value'
    X = data.drop('value', axis=1)
    y = data['value']
    
    # разделяем данные на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

data_path = '../data/prepared_co2_emissions.csv'

X_train, X_test, y_train, y_test = load_and_split_data(data_path)

# сохраняем обучающий и тестовый наборы данных
X_train.to_csv('../data/X_train.csv', index=False)
y_train.to_csv('../data/y_train.csv', index=False)
X_test.to_csv('../data/X_test.csv', index=False)
y_test.to_csv('../data/y_test.csv', index=False)
import pandas as pd

def load_and_split_data(data_path):
    data = pd.read_csv(data_path)
    data['year'] = pd.to_datetime(data['year'])
    data = data.sort_values(by=['country_code', 'year'])
    
    # создаем словарь для хранения обучающих и тестовых данных по каждой стране
    train_data = {}
    test_data = {}
    
    # проходим по каждой стране и разделяем данные на обучающую и тестовую выборки
    for country in data['country_code'].unique():
        country_data = data[data['country_code'] == country]
        
        split_point = int(len(country_data) * 0.8)
        
        #добавляем данные в словари
        train_data[country] = country_data[:split_point]
        test_data[country] = country_data[split_point:]
    
    # конвертируем словари обратно в DataFrame
    train_df = pd.concat(train_data.values(), ignore_index=True)
    test_df = pd.concat(test_data.values(), ignore_index=True)
    
    X_train = train_df.drop('value', axis=1)
    y_train = train_df['value']
    X_test = test_df.drop('value', axis=1)
    y_test = test_df['value']
    
    return X_train, X_test, y_train, y_test

data_path = '../data/prepared_co2_emissions.csv'

X_train, X_test, y_train, y_test = load_and_split_data(data_path)

X_train.to_csv('../data/X_train.csv', index=False)
y_train.to_csv('../data/y_train.csv', index=False)
X_test.to_csv('../data/X_test.csv', index=False)
y_test.to_csv('../data/y_test.csv', index=False)
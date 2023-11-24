import pandas as pd
import os

current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '..', 'data')

file_path = os.path.join(data_dir, 'co2_emissions_mt_by_country.csv')
df = pd.read_csv(file_path)

# зададим границу разделения на обучающую и тестовую выборки
split_year = 2010

# разделение данных на обучающий и тестовый наборы
train = df[df['year'] <= split_year]
test = df[df['year'] > split_year]

# сохранение обучающего и тестового наборов данных
train_file_path = os.path.join(data_dir, 'train_set.csv')
test_file_path = os.path.join(data_dir, 'test_set.csv')

train.to_csv(train_file_path, index=False)
test.to_csv(test_file_path, index=False)
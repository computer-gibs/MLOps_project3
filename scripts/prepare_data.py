import pandas as pd
import os

current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '..', 'data')

file_path = os.path.join(data_dir, 'co2_emissions_kt_by_country.csv')
df = pd.read_csv(file_path)

#проверим наличие пропущенных значений
print("Количество пропущенных значений:")
print(df.isnull().sum())

#удалим строки с пропущенными значениями, если они есть
df.dropna(inplace=True)

#преобразуем значения CO2 из килотонн в мегатонны для удобства
df['value'] = df['value'] / 1000.0

output_file_path = os.path.join(data_dir, 'co2_emissions_mt_by_country.csv')
df.to_csv(output_file_path, index=False)
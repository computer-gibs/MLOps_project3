import pandas as pd

def prepare_data(file_path):
    data = pd.read_csv(file_path)
    
    # убедимся, что данные о годах представлены в правильном формате
    data['year'] = pd.to_datetime(data['year'], format='%Y')
    
    # очистим данные от возможных нулевых или отрицательных значений выбросов
    data = data[data['value'] > 0]
    
    # возвращаем подготовленный набор данных
    return data

# путь к файлу данных
file_path = 'MLOps_project3/data/co2_emissions_kt_by_country.csv'

# подготовка данных
prepared_data = prepare_data(file_path)

# сохраняем подготовленные данные в новый файл
prepared_data.to_csv('MLOps_project3/data/prepared_co2_emissions.csv', index=False)
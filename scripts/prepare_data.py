import pandas as pd

def prepare_data(file_path):
    data = pd.read_csv(file_path)
    
    data['year'] = pd.to_datetime(data['year'], format='%Y')
    
    # очистка данных от нулевых или отрицательных значений
    data = data[data['value'] > 0]
    
    # группируем данные по странам и сортируем внутри каждой группы по годам
    grouped = data.groupby('country_code')
    prepared_data_list = []
    
    for country, group in grouped:
        group = group.sort_values('year')
        
        # Проверка на последовательность годов
        group.set_index('year', inplace=True)
        group = group.asfreq('YS')
        
        # group['value'] = group['value'].interpolate()
        
        prepared_data_list.append(group.reset_index())

    prepared_data = pd.concat(prepared_data_list, ignore_index=True)
    return prepared_data

file_path = '../data/co2_emissions_kt_by_country.csv'

prepared_data = prepare_data(file_path)

prepared_data.to_csv('../data/prepared_co2_emissions.csv', index=False)
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# инициализация API
api = KaggleApi()
api.authenticate()

# путь для сохранения датасета
path_to_save = os.path.join(os.path.dirname(__file__), '..', 'data')

# создаем директорию, если она не существует
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

# скачиваем датасет и распаковываем его
api.dataset_download_files('ulrikthygepedersen/co2-emissions-by-country', path=path_to_save, unzip=True)
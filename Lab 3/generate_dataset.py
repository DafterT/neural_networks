# %%
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# %%
url = 'https://dnd.su/bestiary/'
response = requests.get(url)
# %%
if response.status_code != 200:
    print('Ошибка при получении страницы:', response.status_code)
# %%
soup = BeautifulSoup(response.text, 'html.parser')
grid_element = soup.find('div', class_='grid-4_lg-3_md-2_xs-1')
df = pd.DataFrame(columns=['name', 'link', 'ac', 'health',
                           'strength', 'dexterity',
                           'constitution', 'intelligence',
                           'wisdom', 'charisma'])
df['name'] = [i.find(
    'div', class_='list-item-title').text for i in grid_element.find_all('a')]
df['link'] = [i['href'] for i in grid_element.find_all('a')]
df.head()
# %%
df.info()
# %%
digit_pattern = re.compile("[\+-]?\d+$")


def get_integer(data, pattern=' '):
    for x in re.split(pattern, data.text):
        if digit_pattern.match(x):
            yield x


def get_data_from_url(url, index):
    ready = True
    while ready:
        try:
            response = requests.get(f'https://dnd.su{url}')
            if response.status_code != 200:
                print('Ошибка при получении страницы:', response.status_code)
            soup = BeautifulSoup(response.text, 'html.parser')
            enemy_data = soup.find(
                'ul', class_='params card__article-body').find_all('li')
            ready = False
        except:
            from time import sleep
            print(f'Error in {url}')
            sleep(5)

    df['ac'][index] = int(next(get_integer(enemy_data[1])))
    df['health'][index] = int(next(get_integer(enemy_data[2])))
    stat_generator = get_integer(enemy_data[4], pattern='[\(\) ]')
    df['strength'][index] = int(next(stat_generator))
    df['dexterity'][index] = int(next(stat_generator))
    df['constitution'][index] = int(next(stat_generator))
    df['intelligence'][index] = int(next(stat_generator))
    df['wisdom'][index] = int(next(stat_generator))
    df['charisma'][index] = int(next(stat_generator))


for row in df.itertuples():
    try:
        if row[0] % 10 == 0:
            print(row[0])

        row_now = row[2]
        get_data_from_url(row[2], row[0])
    except Exception:
        print(f'Skipped {row_now}')

# %%
df.info()
# %%
df_not_null = df.dropna(how='any', axis=0).reset_index(drop=True)
df_not_null.info()
# %%
df_not_null.to_csv('./my_data.csv', sep=',', index=True, encoding='utf-8')
# %%
df_data = df_not_null[df_not_null.columns[2:]]
# %%


def normalize(column):
    return (column - column.min()) / ((column.max() - column.min()))


df_data = df_data.apply(normalize, axis=0)
df_data.head()
# %%
df_not_null.iloc[:, 2:] = df_data
df_not_null.head()
# %%
df_not_null.to_csv('./my_data_norm.csv', sep=',', index=True, encoding='utf-8')

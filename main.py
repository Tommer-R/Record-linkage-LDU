import pandas as pd
from time import time
from datetime import datetime, timedelta

scores = pd.read_csv('scores.csv')

thresholds = {'email': 0,
              'company_name': 0,
              'group': 0,
              'phone': 0,
              'city': 0,
              'state': 0,
              'zip': 0,
              'country': 0,
              'name': 0,
              'address': 0,
              }

multipliers = {'email': 1,
               'company_name': 1,
               'group': 1,
               'phone': 1,
               'city': 1,
               'state': 1,
               'zip': 1,
               'country': 1,
               'name': 1,
               'address': 1,
               }


def score(row):
    res = 0
    for col in row.index:
        if col in thresholds.keys() and row[col] >= thresholds[col]:
            res += row[col] * multipliers[col]
    return res


def calc_eta(df, test_size=1000):
    full_length = len(df)
    print(full_length)
    s_time = time()
    df['temp'] = df.loc[:test_size].apply(lambda x: score(x), axis=1)
    delta = time() - s_time
    print(delta)
    eta = timedelta(seconds=int(delta / test_size * full_length))
    now = datetime.now()
    time_eta = (datetime.now() + eta).replace(microsecond=0)
    if now.day == time_eta.day and now.month == time_eta.month:
        time_eta = str(time_eta)[11:]
    print(f'ETA: {eta} | done at: {time_eta}')


def calc_scores(df, column_name='score'):
    df[column_name] = df.apply(lambda x: score(x), axis=1)
    return df


calc_eta(scores)
scores = calc_scores(scores)

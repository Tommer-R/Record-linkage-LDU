import numpy as np
import pandas as pd
from time import time
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

hw = pd.read_pickle('data/processed/hw_processed.pkl')
lda = pd.read_pickle('data/processed/lda_processed.pkl')
hw_raw = pd.read_csv('data/raw/HeroWeb Accounts.csv', delimiter=';')
lda_raw = pd.read_csv('data/raw/Priority Customers.csv', delimiter=';')
scores = pd.read_pickle('data/generated/scores.pkl')


def validate_strings(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else x)
        df[col] = df[col].apply(lambda x: x.removesuffix('.0') if pd.notnull(x) and x.endswith('.0') else x)
    return df


def plot_distribution(x, title='No title', range_=None):
    face_color = '#EAEAEA'
    color_bars = '#3475D0'
    txt_color1 = '#252525'
    txt_color2 = '#004C74'

    fig, ax = plt.subplots(1, figsize=(20, 6), facecolor=face_color)
    ax.set_facecolor(face_color)
    n, bins, patches = plt.hist(x, color=color_bars, bins=10, range=range_)

    # grid
    minor_locator = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', color=face_color, lw=0.5)
    xticks = [(bins[idx + 1] + value) / 2 for idx, value in enumerate(bins[:-1])]
    xticks_labels = [f"{round(value, 1)}-{round(bins[idx + 1], 1)}" for idx, value in enumerate(bins[:-1])]
    plt.xticks(xticks, labels=xticks_labels, c=txt_color1, fontsize=13)

    # remove major and minor ticks from the x axis, but keep the labels
    ax.tick_params(axis='x', which='both', length=0)
    # remove y ticks
    plt.yticks([])

    text_str = '\n'.join((
        f'mean={round(x.mean(), 1)}',
        f'median={round(float(np.median(x)), 1)}',
        f'std={round(x.std(), 1)}',))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor=face_color, alpha=0.5)

    # place a text box in upper left in axes coordinates
    ax.text(0.85, 0.95, text_str, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='center', bbox=props)

    # Hide the right and top spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for idx, value in enumerate(n):
        if value > 0:
            plt.text(xticks[idx], value + 10, "{:,}".format(value), ha='center', fontsize=16, c=txt_color1)
    plt.title(title, loc='left', fontsize=20, c=txt_color1)
    plt.xlabel('Count', c=txt_color2, fontsize=14)
    plt.ylabel('Score', c=txt_color2, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'plots/{title}.png', facecolor=face_color)
    plt.clf()
    print(f'saved plot: "{title}.png"')


def plot_all(df):
    plot_distribution(df['score'], 'score', range_=(2, df['score'].max()))
    plot_distribution(df['total'], 'score', range_=(2, df['score'].max()))

    for col in df.columns:
        if col not in {'total', 'score', 'index1', 'index2'}:
            plot_distribution(df[col], col)


hw = validate_strings(hw)
lda = validate_strings(lda)
hw_raw = validate_strings(hw_raw)
lda_raw = validate_strings(lda_raw)

thresholds = {'email': 0.5,
              'company_name': 0,
              'group': 0.25,
              'phone': 0.25,
              'city': 0.5,
              'state': 0.5,
              'zip': 0,
              'country': 0.5,
              'name': 0,
              'address': 0,
              }

multipliers = {'email': 1,
               'company_name': 1,
               'group': 0.1,
               'phone': 1,
               'city': 0.5,
               'state': 0.25,
               'zip': 1,
               'country': 0.25,
               'name': 1,
               'address': 0.8,
               }


def score(row):
    res = 0
    for col in row.index:
        if col in thresholds.keys() and row[col] >= thresholds[col]:
            res += row[col] * multipliers[col]
    return res


def calc_eta(df, test_size=10000):
    slice_ = df.loc[:test_size].copy()
    full_length = len(df)
    s_time = time()

    slice_['temp'] = slice_.apply(lambda x: score(x), axis=1)
    delta = time() - s_time
    eta = timedelta(seconds=int(delta / test_size * full_length))
    now = datetime.now()
    time_eta = (datetime.now() + eta).replace(microsecond=0)
    if now.day == time_eta.day and now.month == time_eta.month:
        time_eta = str(time_eta)[11:]
    print(f'ETA: {eta} | done at: {time_eta}')


def calc_scores(df, column_name='score', save=False):
    df[column_name] = df.apply(lambda x: score(x), axis=1)
    if save:
        scores.to_pickle('data/generated/scores.pkl')
    return df


calc_eta(scores)
scores = calc_scores(scores, save=False)

description = scores.describe()

plot_all(scores)


def match_filter(min_score, full_match_at=1, all_match: list[str] = None, some_match: list[str] = None):
    mask = scores['score'] >= min_score

    if all_match is not None:
        for col in all_match:
            mask = mask & scores[col] >= full_match_at

    if some_match is not None:
        temp_masks = []
        for col in some_match:
            temp_masks.append(scores[col] >= full_match_at)
        temp_mask = temp_masks[0]
        for m in temp_masks[1:]:
            temp_mask = temp_mask | m

        mask = mask & temp_mask

    return mask


matches = scores[match_filter(5)]


def evaluate_matches(df):
    verified = []
    verified_false = []
    non_verified = []
    for i in df.index:
        match = df.loc[i]
        hw_index = match['index1']
        lda_index = match['index2']
        if pd.notnull(lda_raw.loc[lda_index, 'HW Account']):
            if lda_raw.loc[lda_index, 'HW Account'] == hw_raw.loc[hw_index, 'account_id']:
                verified.append(i)
            else:
                verified_false.append(i)
        else:
            non_verified.append(i)

    num_true_matches = len(lda_raw.loc[pd.notnull(lda_raw['HW Account'])])
    print('====================== REPORT ======================')
    print(f'total matches: {len(df)}')
    print(f'total true matches: {num_true_matches}')
    print(f'verified matches: {len(verified)}')
    print(f'verified false matches: {len(verified_false)}')
    print(f'non verified matches: {len(non_verified)}')
    print(f'of total true matches found: {round(len(verified)/num_true_matches*100, 2)}%')
    print(f'verified share: {round(len(verified)/len(df)*100, 2)}%')
    print(f'false share: {round(len(verified_false) / len(df) * 100, 2)}%')




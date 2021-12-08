import numpy as np
import pandas as pd
from time import time
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

hw = pd.read_pickle('data/processed/hw_processed.pkl')
lda = pd.read_pickle('data/processed/lda_processed.pkl')
scores = pd.read_pickle('data/generated/scores.pkl')


def validate_strings(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else x)
        df[col] = df[col].apply(lambda x: x.removesuffix('.0') if pd.notnull(x) and x.endswith('.0') else x)
    return df


hw = validate_strings(hw)
lda = validate_strings(lda)


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
    s_time = time()
    df['temp'] = df.loc[:test_size].apply(lambda x: score(x), axis=1)
    delta = time() - s_time
    eta = timedelta(seconds=int(delta / test_size * full_length))
    now = datetime.now()
    time_eta = (datetime.now() + eta).replace(microsecond=0)
    if now.day == time_eta.day and now.month == time_eta.month:
        time_eta = str(time_eta)[11:]
    print(f'ETA: {eta} | done at: {time_eta}')


def calc_scores(df, column_name='score'):
    df[column_name] = df.apply(lambda x: score(x), axis=1)
    return df


# calc_eta(scores)
# scores = calc_scores(scores)


def plot_distribution(x, title='No title', range_=None):
    facecolor = '#EAEAEA'
    color_bars = '#3475D0'
    txt_color1 = '#252525'
    txt_color2 = '#004C74'

    fig, ax = plt.subplots(1, figsize=(20, 6), facecolor=facecolor)
    ax.set_facecolor(facecolor)
    n, bins, patches = plt.hist(x, color=color_bars, bins=10, range=range_)

    # grid
    minor_locator = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', color=facecolor, lw=0.5)
    xticks = [(bins[idx + 1] + value) / 2 for idx, value in enumerate(bins[:-1])]
    xticks_labels = [f"{round(value, 1)}-{round(bins[idx + 1], 1)}" for idx, value in enumerate(bins[:-1])]
    plt.xticks(xticks, labels=xticks_labels, c=txt_color1, fontsize=13)

    # remove major and minor ticks from the x axis, but keep the labels
    ax.tick_params(axis='x', which='both', length=0)
    # remove y ticks
    plt.yticks([])

    text_str = '\n'.join((
        f'mean={round(x.mean(), 1)}',
        f'median={round(np.median(x), 1)}',
        f'std={round(x.std(), 1)}',))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor=facecolor, alpha=0.5)

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
    plt.savefig(f'plots/{title}.png', facecolor=facecolor)
    plt.clf()
    print(f'saved plot: "{title}.png"')


# plot_distribution(scores['score'], 'score', range_=(2, scores['score'].max()))
# plot_distribution(scores['total'], 'score', range_=(2, scores['score'].max()))

for col in scores.columns:
    if col not in {'total', 'score', 'index1', 'index2'}:
        # plot_distribution(scores[col], col)
        pass


# scores.to_pickle('data/generated/scores.pkl')

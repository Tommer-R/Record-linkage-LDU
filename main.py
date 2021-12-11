import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

tqdm.pandas()

hw = pd.read_pickle('data/processed/hw_processed.pkl')
lda = pd.read_pickle('data/processed/lda_processed.pkl')
hw_raw = pd.read_csv('data/raw/HeroWeb Accounts.csv', delimiter=';')
lda_raw = pd.read_csv('data/raw/Priority Customers.csv', delimiter=';')
scores = pd.read_pickle('data/generated/scores.pkl')

scores = scores[:]


def evaluate_matches(df, plot=True):
    print('evaluating matches')
    verified_ = []
    verified_false_ = []
    non_verified_ = []
    for i in tqdm(df.index):
        match_ = df.loc[i]
        hw_index = match_['index1']
        lda_index = match_['index2']
        if pd.notnull(lda_raw.loc[lda_index, 'HW Account']):
            if lda_raw.loc[lda_index, 'HW Account'] == hw_raw.loc[hw_index, 'account_id']:
                verified_.append(i)
            else:
                verified_false_.append(i)
        else:
            non_verified_.append(i)

    num_true_matches = len(lda_raw.loc[pd.notnull(lda_raw['HW Account'])])
    percent_verified = round(len(verified_) / len(df) * 100, 2)
    percent_false = round(len(verified_false_) / len(df) * 100, 2)
    percent_non_verified = 100 - percent_false - percent_verified

    print('\n====================== REPORT ======================')
    print(f'total matches: {len(df)}')
    print(f'total true matches: {num_true_matches}')
    print(f'verified matches: {len(verified_)}')
    print(f'verified false matches: {len(verified_false_)}')
    print(f'non verified matches: {len(non_verified_)}')
    print(f'of total true matches found: {round(len(verified_) / num_true_matches * 100, 2)}%')
    print(f'verified share: {percent_verified}%')
    print(f'false share: {percent_false}%')
    print('====================================================\n')

    if plot:
        face_color = '#EAEAEA'

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'True', 'non-verified', 'False'
        sizes = [percent_verified, percent_non_verified, percent_false]
        explode = (0, 0, 0)  # only "explode" one slice

        fig1, ax1 = plt.subplots(1, figsize=(10, 6), facecolor=face_color)
        ax1.set_facecolor(face_color)
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%',
                shadow=False, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        text_str = '\n'.join((
            f'total matches: {len(df)}',
            f'total true matches: {num_true_matches}',
            f'verified matches: {len(verified_)}',
            f'verified false matches: {len(verified_false_)}',
            f'non verified matches: {len(non_verified_)}'))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor=face_color, alpha=0.5)

        # place a text box in upper left in axes coordinates
        ax1.text(0.05, 1, text_str, transform=ax1.transAxes, fontsize=14,
                 verticalalignment='top', horizontalalignment='center', bbox=props)

        plt.title('Matches evaluation')
        plt.savefig(f'plots/Evaluation.png', facecolor=face_color)
        plt.show()
        plt.clf()

    return df.loc[verified_], df.loc[verified_false_], df.loc[non_verified_]


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
    x_ticks = [(bins[idx + 1] + value) / 2 for idx, value in enumerate(bins[:-1])]
    x_ticks_labels = [f"{round(value, 1)}-{round(bins[idx + 1], 1)}" for idx, value in enumerate(bins[:-1])]
    plt.xticks(x_ticks, labels=x_ticks_labels, c=txt_color1, fontsize=13)

    # remove major and minor ticks from the x-axis, but keep the labels
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
            plt.text(x_ticks[idx], value + 10, "{:,}".format(value), ha='center', fontsize=16, c=txt_color1)
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


def plot_combined(df, min_=3):
    face_color = '#EAEAEA'
    color_bars = '#3475D0'
    txt_color1 = '#252525'
    txt_color2 = '#004C74'

    columns = [c for c in df.columns if c not in {'index1', 'index2'}][::-1]

    fig, axs = plt.subplots(len(columns), figsize=(20, len(columns) * 6), facecolor=face_color)
    for i, col in enumerate(columns):
        x = df[col]
        axs[i].set_facecolor(face_color)
        if col in {'total', 'score'}:
            r = (min_, x.max())
        else:
            r = None
        n, bins, patches = axs[i].hist(x, color=color_bars, bins=10, range=r)

        # grid
        minor_locator = AutoMinorLocator(2)
        plt.gca().xaxis.set_minor_locator(minor_locator)
        axs[i].grid(which='minor', color=face_color, lw=0.5)
        xticks = [(bins[idx + 1] + value) / 2 for idx, value in enumerate(bins[:-1])]
        xticks_labels = [f"{round(value, 1)}-{round(bins[idx + 1], 1)}" for idx, value in enumerate(bins[:-1])]
        axs[i].set_xticks(xticks, labels=xticks_labels, c=txt_color1, fontsize=13)

        # remove major and minor ticks from the x axis, but keep the labels
        axs[i].tick_params(axis='x', which='both', length=0)
        # remove y ticks
        axs[i].set_yticks([])

        text_str = '\n'.join((
            f'mean={round(x.mean(), 2)}',
            f'median={round(float(np.median(x)), 2)}',
            f'std={round(x.std(), 2)}',))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor=face_color, alpha=0.5)

        # place a text box in upper left in axes coordinates
        axs[i].text(0.85, 0.95, text_str, transform=axs[i].transAxes, fontsize=14,
                    verticalalignment='top', horizontalalignment='center', bbox=props)

        # Hide the right and top spines
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)

        for idx, value in enumerate(n):
            if value > 0:
                axs[i].text(xticks[idx], value + 10, "{:,}".format(value), ha='center', fontsize=16, c=txt_color1)
        axs[i].set_title(col, loc='left', fontsize=20, c=txt_color1)
        axs[i].set_xlabel('Count', c=txt_color2, fontsize=14)
        axs[i].set_ylabel('Score', c=txt_color2, fontsize=14)
        plt.tight_layout()

    plt.savefig(f'plots/combined_plot.png', facecolor=face_color)
    plt.clf()
    print(f'saved plot: "combined_plot.png"')


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
               'address': 1,
               }


def score(row):
    res = 0
    for col in row.index:
        if col in thresholds.keys() and row[col] >= thresholds[col]:
            res += row[col] * multipliers[col]
    return res


def calc_scores(df, column_name='score', func=None, vectorize: bool = True, save: bool = False):
    print('started finale score calculation')
    if vectorize:
        df[column_name] = (df['email'] * multipliers['email'] * (df['email'] >= thresholds['email'])) + \
                          (df['company_name'] * multipliers['company_name'] * (
                                  df['company_name'] >= thresholds['company_name'])) + \
                          (df['group'] * multipliers['group'] * (df['group'] >= thresholds['group'])) + \
                          (df['phone'] * multipliers['phone'] * (df['phone'] >= thresholds['phone'])) + \
                          (df['city'] * multipliers['city'] * (df['city'] >= thresholds['city'])) + \
                          (df['state'] * multipliers['state'] * (df['state'] >= thresholds['state'])) + \
                          (df['zip'] * multipliers['zip'] * (df['zip'] >= thresholds['zip'])) + \
                          (df['country'] * multipliers['country'] * (df['country'] >= thresholds['country'])) + \
                          (df['name'] * multipliers['name'] * (df['name'] >= thresholds['name'])) + \
                          (df['address'] * multipliers['address'] * (df['address'] >= thresholds['address']))
    else:
        if func is None:
            func = score
        df[column_name] = df.progress_apply(lambda x: func(x), axis=1)

    if save:
        df.to_pickle('data/generated/scores.pkl')
        print('saved new scores in: data/generated/scores.pkl')
    print('finished finale score calculation')

    return df


def match_filter(df, min_score, full_match_at=1, all_match: list[str] = None, some_match: list[str] = None):
    mask = df['score'] >= min_score

    if all_match is not None:
        for col in all_match:
            mask = mask & df[col] >= full_match_at

    if some_match is not None:
        temp_masks = []
        for col in some_match:
            temp_masks.append(df[col] >= full_match_at)
        temp_mask = temp_masks[0]
        for m in temp_masks[1:]:
            temp_mask = temp_mask | m

        mask = mask & temp_mask

    return mask


scores = calc_scores(scores, func=score, vectorize=True, save=False)  # finale score through score formula

description = scores.describe()


# plot_all(scores)

# plot_combined(scores)


# matches = scores[match_filter(scores, 5)]

def match(df):
    masks = []

    name = df['name'] >= 0.8
    company = df['company_name'] >= 0.9
    email = df['email'] >= 0.9
    phone = df['phone'] >= 0.9

    address = df['address'] >= 0.9

    city = df['city'] >= 1
    state = df['state'] >= 1
    zip_ = df['zip'] >= 1
    country = df['country'] >= 1

    score1 = df['score'] >= 5.9
    score2 = df['score'] >= 3.4

    masks.append(score1)
    masks.append(score2 & ((name | company) & (email | phone)))
    masks.append(score2 & city & state & zip_ & country & address)

    print(f"score mask: {len(df.loc[masks[0]])} | unique: {len(df.loc[masks[0] & ~ (masks[1] | masks[2])])}")
    print(f"contact mask: {len(df.loc[masks[1]])} | unique: {len(df.loc[masks[1] & ~ (masks[0] | masks[2])])}")
    print(f"address mask: {len(df.loc[masks[2]])} | unique: {len(df.loc[masks[2] & ~ (masks[1] | masks[0])])}")

    mask = masks[0]
    for m in masks:
        mask = mask | m

    return df.loc[mask]


matches = match(scores)
print(f'generated matches: {len(matches)}')

verified, verified_false, non_verified = evaluate_matches(matches)

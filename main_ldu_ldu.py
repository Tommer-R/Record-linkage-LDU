import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

tqdm.pandas()

lda = pd.read_pickle('data/processed/lda_processed.pkl')
lda_raw = pd.read_pickle('data/raw/lda_raw.pkl')
scores = pd.read_pickle('data/generated/scores_ldu_ldu.pkl')

lda.drop(columns=['hw id'], inplace=True)
lda_raw.drop(columns=['hw id'], inplace=True)


def group_matches(matches_):
    print(f'grouping {len(matches_)} matches')
    indexes1 = tuple(matches_['index1'])
    indexes2 = tuple(matches_['index2'])
    groups_ = [[indexes1[0], indexes2[0]]]
    for idx1, idx2 in zip(indexes1[1:], indexes2[1:]):

        i = 0
        while True:

            try:
                group = groups_[i]
            except IndexError:
                break

            # print(groups)
            if idx1 in group and idx2 in group:
                i += 1
            elif idx1 in group:
                group.append(idx2)
            elif idx2 in group:
                group.append(idx1)
            else:
                if i + 1 == len(groups_):
                    groups_.append([idx1, idx2])
                i += 1
    print(f'created {len(groups_)} groups')
    return groups_


def matches_to_df(df: pd.DataFrame, groups_: list[list[int]]):
    columns = list(df.columns)
    res = {}
    for c in columns:
        res[c] = []

    for group in groups_:
        for idx in group:
            row = df.loc[idx]
            for col, val in zip(tuple(row.index), tuple(row.values)):
                res[col].append(val)
            for col in [item for item in columns if item not in list(row.index)]:
                res[col].append('')
        for col in res.keys():
            res[col].append('')

    return pd.DataFrame(data=res)


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

    plt.savefig(f'plots/combined_plot_ldu_ldu.png', facecolor=face_color)
    plt.clf()
    print(f'saved plot: "combined_plot_ldu_ldu.png"')


def score(row):
    res = 0
    for col in row.index:
        if col in thresholds.keys() and row[col] >= thresholds[col]:
            res += row[col] * multipliers[col]
    return res


def calc_combined_scores(df, column_name='score', func=None, vectorize: bool = True, save: bool = False):
    print('started finale score calculation')
    if vectorize:
        df[column_name] = (df['email'] * multipliers['email'] * (df['email'] >= thresholds['email'])) + \
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


# make sure all datatypes are correct
lda = validate_strings(lda)
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
scores = calc_combined_scores(scores, func=score, vectorize=True, save=False)  # finale score through score formula


# description = scores.describe()

# plot_all(scores)

# plot_combined(scores)


def match(df):
    masks = []

    masks.append(df['score'] >= 4.9)

    masks.append((df['score'] >= 3.4) & ((df['name'] >= 0.7) &
                                         ((df['email'] >= 0.8) | (df['phone'] >= 0.9) | (df['fax'] >= 0.8))))

    masks.append((df['score'] >= 3.4) & (df['city'] >= 0.9) & (df['state'] >= 0.9) & (df['zip'] >= 0.9) &
                 (df['country'] >= 0.9) & (df['address'] >= 1))

    masks.append((df['name'] == 1) | (df['email'] == 1))

    print(f"\nscore mask: {len(df.loc[masks[0]])} | "
          f"unique: {len(df.loc[masks[0] & ~ (masks[1] | masks[2] | masks[3])])}",
          f"\ncontact mask: {len(df.loc[masks[1]])} | "
          f"unique: {len(df.loc[masks[1] & ~ (masks[0] | masks[2] | masks[3])])}",
          f"\naddress mask: {len(df.loc[masks[2]])} | "
          f"unique: {len(df.loc[masks[2] & ~ (masks[1] | masks[0] | masks[3])])}",
          f"\nexact mask: {len(df.loc[masks[3]])} | "
          f"unique: {len(df.loc[masks[3] & ~ (masks[1] | masks[0] | masks[2])])}")

    mask = masks[0]
    for m in masks:
        mask = mask | m

    return df.loc[mask]


matches = match(scores)  # choose what is considered as a match
print(f'generated matches: {len(matches)}')

# verified, verified_false, non_verified = evaluate_matches(matches, plot=False)

groups = group_matches(matches)  # create groups from all matches

# convert groups list to  dataframe
grouped_matches = matches_to_df(lda, groups)
grouped_matches_raw = matches_to_df(lda_raw, groups)

# prepare dataframes to presentation
grouped_matches.replace(np.nan, '', inplace=True)
grouped_matches_raw.replace(np.nan, '', inplace=True)


with pd.ExcelWriter('data/generated/matches_ldu_ldu.xlsx') as writer:  # save to excel
    grouped_matches_raw.to_excel(writer, sheet_name='all')

print('saved results')

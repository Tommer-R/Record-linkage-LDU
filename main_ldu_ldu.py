import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

tqdm.pandas()

# load relevant datasets
lda = pd.read_pickle('data/generated/lda_processed.pkl')
lda_raw = pd.read_pickle('data/raw/lda_raw.pkl')
scores = pd.read_pickle('data/generated/scores_ldu_ldu.pkl')

lda.drop(columns=['hw id'], inplace=True)
lda_raw.drop(columns=['hw id'], inplace=True)


def validate_strings(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else x)
        df[col] = df[col].apply(lambda x: x.removesuffix('.0') if pd.notnull(x) and x.endswith('.0') else x)
    return df


def group_matches(matches_):
    """
    group all related selected matches in to groups that represent the same client.
    grouping assumes that if: A == B and: B == C then:  [A, B, C]  are all the same client.
    """
    print(f'grouping {len(matches_)} matches')
    indexes1 = tuple(matches_['index1'])  # list of indexes
    indexes2 = tuple(matches_['index2'])  # list of matching indexes
    groups_ = [[indexes1[0], indexes2[0]]]  # first group
    links_ = [[indexes1[0], indexes2[0]]]
    assert len(indexes1) == len(indexes2)

    for idx1, idx2 in zip(indexes1[1:], indexes2[1:]):  # iterate through matching indexes
        links_.append([idx1, idx2])
        i = 0  # group index
        while True:
            try:
                group = groups_[i]  # current group
            except IndexError:
                break

            if idx1 in group and idx2 in group:  # match already exists in group
                i += 1  # next group
                break
            elif idx1 in group:  # first index in group but second isn't
                group.append(idx2)  # add the second index
                break
            elif idx2 in group:  # second index in group but first isn't
                group.append(idx1)  # add the first index
                break
            else:  # both indexes are not in group
                if i + 1 == len(groups_):  # last group
                    groups_.append([idx1, idx2])  # add match as new group
                    break
                i += 1  # next group

    print(f'created {len(groups_)} groups')
    return groups_, links_


def validate_groups(groups_: list, print_dupes: bool):
    ids = []
    dupes = []
    lengths = []

    for g in groups_:
        lengths.append(len(g))
        for a in g:
            if a in ids:
                dupes.append(a)
            else:
                ids.append(a)

    if len(dupes) == 0:
        print('validated groups, no duplicates')
    else:
        if print_dupes:
            print(f'dupes: {dupes}')
        print(f'duplications found inside groups: {len(dupes)}')
        print(f'max length group: {np.max(lengths)}')
    lengths_df = pd.Series(lengths)
    return lengths_df.describe()


def groups_to_df(df: pd.DataFrame, groups_: list[list[int]], links_):
    """
    convert list of groups to a dataframe, in the dataframe each row is an original record and all sequential rows
    are the same group, the same client. between groups are rows of empty strings
    this is meant for easy manual examination and not for further processing
    """
    columns = list(df.columns)  # result df columns
    res = {'match': []}  # use dict for better performance
    for c in columns:  # init dict structure
        res[c] = []

    for group in groups_:
        for idx in group:
            row = df.loc[idx]  # row to add to res dict
            matching_indexes = set()
            for n in links_:  # add matches ids
                if idx in n:
                    l2 = n[:]
                    l2.remove(idx)
                    matching_indexes.add(df['id'][l2[0]])  # add id

            for col, val in zip(tuple(row.index), tuple(row.values)):
                res[col].append(val)  # add value to dict
            if len(matching_indexes) == 1:
                res['match'].append(str(list(matching_indexes)[0]))
            elif len(matching_indexes) >= 1:
                res['match'].append(', '.join([str(elem) for elem in list(matching_indexes)]))  # show all matches
            else:
                res['match'].append('')  # empty string

            for col in [item for item in columns if item not in list(row.index)]:
                res[col].append('')  # fill missing values with empty strings
        for col in res.keys():  # add row of empty string after group
            res[col].append('')

    return pd.DataFrame(data=res)  # convert res dict to dataframe


def plot_combined(df, min_=3):
    """create joined distribution plot for all columns"""
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


def calc_combined_scores(df: pd.DataFrame, column_name='score', save: bool = False):
    """
    combine all value scores of a link using the formula:
    combined score += value * multiplier if value >= threshold

    using vectorized calculation for performance reasons
    """
    print('started finale score calculation')

    # don`t add score if it is smaller than that
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

    # multiply score by this when adding it
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

    df[column_name] = (df['email'] * multipliers['email'] * (df['email'] >= thresholds['email'])) + \
                      (df['group'] * multipliers['group'] * (df['group'] >= thresholds['group'])) + \
                      (df['phone'] * multipliers['phone'] * (df['phone'] >= thresholds['phone'])) + \
                      (df['city'] * multipliers['city'] * (df['city'] >= thresholds['city'])) + \
                      (df['state'] * multipliers['state'] * (df['state'] >= thresholds['state'])) + \
                      (df['zip'] * multipliers['zip'] * (df['zip'] >= thresholds['zip'])) + \
                      (df['country'] * multipliers['country'] * (df['country'] >= thresholds['country'])) + \
                      (df['name'] * multipliers['name'] * (df['name'] >= thresholds['name'])) + \
                      (df['address'] * multipliers['address'] * (df['address'] >= thresholds['address']))

    if save:
        df.to_pickle('data/generated/scores.pkl')
        print('saved new scores in: data/generated/scores.pkl')
    print('finished finale score calculation')

    return df


# make sure all datatypes are correct
lda = validate_strings(lda)
lda_raw = validate_strings(lda_raw)

scores = calc_combined_scores(scores, save=False)  # finale score through score formula


# scores_description = scores.describe()

# plot_combined(scores)


def match(df):
    """
    decide which comparisons are considered to be a match.

    using different conditions to define what is a match, a match must meet at least one of them.
    """
    masks = []

    masks.append(df['score'] >= 4.9)  # combined score

    # combined score is not too small and, name matches and, email or phone or fax matches
    masks.append((df['score'] >= 3.4) & ((df['name'] >= 0.7) &
                                         ((df['email'] >= 0.8) | (df['phone'] >= 0.9) | (df['fax'] >= 0.8))))

    # combined score is not small and, all location attributes match
    masks.append((df['score'] >= 3.4) & (df['city'] >= 0.9) & (df['state'] >= 0.9) & (df['zip'] >= 0.9) &
                 (df['country'] >= 0.9) & (df['address'] >= 1))

    # name or email or phone is a perfect match
    masks.append((df['name'] == 1) | (df['email'] == 1) | (df['phone'] == 1))

    print(f"\nscore mask: {len(df.loc[masks[0]])} | "
          f"unique: {len(df.loc[masks[0] & ~ (masks[1] | masks[2] | masks[3])])}",
          f"\ncontact mask: {len(df.loc[masks[1]])} | "
          f"unique: {len(df.loc[masks[1] & ~ (masks[0] | masks[2] | masks[3])])}",
          f"\naddress mask: {len(df.loc[masks[2]])} | "
          f"unique: {len(df.loc[masks[2] & ~ (masks[1] | masks[0] | masks[3])])}",
          f"\nexact mask: {len(df.loc[masks[3]])} | "
          f"unique: {len(df.loc[masks[3] & ~ (masks[1] | masks[0] | masks[2])])}\n")

    mask = masks[0]
    for m in masks:  # combine all matches
        mask = mask | m

    return df.loc[mask]


matches = match(scores)  # choose what is considered as a match
print(f'generated matches: {len(matches)}')

# verified, verified_false, non_verified = evaluate_matches(matches, plot=False)

groups, links = group_matches(matches)  # create groups from all matches

length_description = validate_groups(groups, print_dupes=False)

# convert groups list to  dataframe
grouped_matches = groups_to_df(lda, groups, links)
grouped_matches_raw = groups_to_df(lda_raw, groups, links)

# prepare dataframes to presentation
grouped_matches.replace(np.nan, '', inplace=True)
grouped_matches = grouped_matches[['id', 'match', 'name', 'phone', 'fax', 'email', 'group', 'address',
                                   'city', 'state', 'zip', 'country', 'web_site']]
grouped_matches_raw.replace(np.nan, '', inplace=True)
grouped_matches_raw = grouped_matches_raw[['id', 'match', 'name', 'phone', 'fax', 'email', 'group', 'address',
                                           'city', 'state', 'zip', 'country', 'web_site']]

with pd.ExcelWriter('data/generated/matches_ldu_ldu.xlsx') as writer:  # save to excel
    grouped_matches_raw.to_excel(writer, sheet_name='all')

print('saved results')

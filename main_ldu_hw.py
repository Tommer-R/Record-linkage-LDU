import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

tqdm.pandas()

# load relevant datasets
hw = pd.read_pickle('data/generated/hw_processed.pkl')
ldu = pd.read_pickle('data/generated/lda_processed.pkl')
hw_raw = pd.read_pickle('data/raw/hw_raw.pkl')
ldu_raw = pd.read_pickle('data/raw/lda_raw.pkl')
scores = pd.read_pickle('data/generated/scores_ldu_hw.pkl')


def validate_strings(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else x)
        df[col] = df[col].apply(lambda x: x.removesuffix('.0') if pd.notnull(x) and x.endswith('.0') else x)
    return df


def group_matches(matches_: pd.DataFrame):
    """
    group all related selected matches in to groups that represent the same client.
    grouping assumes that if: A == B and: B == C then:  [A, B, C]  are all the same client.
    """
    print(f'grouping {len(matches_)} matches')
    indexes1 = tuple(matches_['index1'])  # list of indexes from first dataframe
    indexes2 = tuple(matches_['index2'])  # list of matching indexes from second dataframe
    groups_ = [([indexes1[0]], [indexes2[0]])]  # first group
    links_ = [(indexes1[0], indexes2[0])]
    assert len(indexes1) == len(indexes2)

    for idx1, idx2 in zip(indexes1[1:], indexes2[1:]):  # iterate through matching indexes
        links_.append((idx1, idx2))
        i = 0  # group index
        while True:
            try:
                group = groups_[i]  # current group
            except IndexError:
                break

            if idx1 in group[0] and idx2 in group[1]:  # match already exists in group
                i += 1  # next group
                break
            elif idx1 in group[0]:  # first index in group but second isn't
                group[1].append(idx2)  # add the second index
                break
            elif idx2 in group[1]:
                group[0].append(idx1)
                break
            else:  # both indexes are not in group
                if i + 1 == len(groups_):  # last group
                    groups_.append(([idx1], [idx2]))  # add match as new group
                    break
                i += 1  # next group

    print(f'created {len(groups_)} groups')
    return groups_, links_


def validate_groups(groups_: list, print_dupes: bool):
    ids1 = []
    ids2 = []
    dupes1 = []
    dupes2 = []
    lengths = []

    for g in groups_:
        lengths.append(len(g[0]) + len(g[1]))
        for a in g[0]:
            if a in ids1:
                dupes1.append(a)
            else:
                ids1.append(a)
        for a in g[1]:
            if a in ids2:
                dupes2.append(a)
            else:
                ids2.append(a)

    if len(dupes1) == 0 and len(dupes2) == 0:
        print('validated groups, no duplicates')
    else:
        if print_dupes:
            print(f'HW dupes: {dupes1}')
            print(f'LDU dupes: {dupes2}')
        print(f'duplications found inside groups: hw-{len(dupes1)}, ldu-{len(dupes2)}')
        print(f'max length group: {np.max(lengths)}')

    lengths_df = pd.Series(lengths)
    return lengths_df.describe()


def groups_to_df(df1: pd.DataFrame, df2: pd.DataFrame, groups_: list[tuple[list[int], list[int]]], links_):
    """
    convert list of groups to a dataframe, in the dataframe each row is an original record and all sequential rows
    are the same group, the same client. between groups are rows of empty strings
    this is meant for easy manual examination and not for further processing
    """
    columns = list(set(list(df1.columns) + list(df2.columns))) + ['source']  # result df columns
    res = {'match': []}  # use dict for better performance
    for c in columns:  # init dict structure
        res[c] = []

    for group in groups_:
        for idx in group[0]:  # add matches from first dataframe
            row = df1.loc[idx]  # row to add to res dict
            matching_indexes = set()
            for n in links_:  # add matches ids
                if idx == n[0]:
                    matching_indexes.add(df2['id'][n[1]])  # add id
            for col, val in zip(tuple(row.index), tuple(row.values)):
                res[col].append(val)  # add value to dict
            for col in [item for item in columns if item not in list(row.index)]:
                res[col].append('')  # fill missing values with empty strings
            res['source'][-1] = 'hw'  # add name of source df
            if len(matching_indexes) == 1:
                res['match'].append(str(list(matching_indexes)[0]))
            elif len(matching_indexes) >= 1:
                res['match'].append(', '.join([str(elem) for elem in list(matching_indexes)]))  # show all matches
            else:
                res['match'].append('')  # empty string
        for idx in group[1]:  # add matches from second dataframe
            row = df2.loc[idx]  # row to add to res dict
            matching_indexes = set()
            for n in links_:  # add matches ids
                if idx == n[1]:
                    matching_indexes.add(df1['id'][n[0]])  # add id
            for col, val in zip(tuple(row.index), tuple(row.values)):
                res[col].append(val)  # add value to dict
            for col in [item for item in columns if item not in list(row.index)]:
                res[col].append('')  # fill missing values with empty strings
            res['source'][-1] = 'ldu'  # add name of source df
            if len(matching_indexes) == 1:
                res['match'].append(str(list(matching_indexes)[0]))
            elif len(matching_indexes) >= 1:
                res['match'].append(', '.join([str(elem) for elem in list(matching_indexes)]))  # show all matches
            else:
                res['match'].append('')  # empty string
        for col in res.keys():  # add row of empty string after group
            res[col].append('')

    return pd.DataFrame(data=res)  # convert res dict to dataframe


# prepare dataframe for easier examination
def to_presentation(df: pd.DataFrame, drop_hw_id: bool):
    sorted_columns = ['source', 'id', 'hw id', 'match', 'name', 'name2', 'company_name', 'email', 'phone', 'phone3', 'phone2',
                      'fax', 'web_site', 'group', 'address', 'zip', 'city', 'state', 'country', 'address2', 'city2',
                      'zip2', 'state2', 'country2', 'address3', 'address4']
    df = df[sorted_columns]  # reorder columns
    if drop_hw_id:
        df = df.drop(columns=['hw id'])
    df.replace(np.nan, '', inplace=True)  # replace all missing values with empty string
    df['source'].replace('ldu', 'LandsDownUnder', inplace=True)  # use the full name instead of short
    df['source'].replace('hw', 'HeroWeb', inplace=True)  # use the full name instead of short
    return df


def separate_groups(df1, df2, groups_):
    """
    separate groups according to pre labeled data to verified and non verified groups
    """
    verified_ = []
    not_verified_ = []

    for group in groups_:
        sorted_ = False
        ids1 = [df1.loc[i, 'id'] for i in group[0]]
        ids2 = [df2.loc[i, 'id'] for i in group[1]]
        ids3 = [df2.loc[i, 'hw id'] for i in group[1]]

        ids3 = list(filter(lambda a: pd.notnull(a), ids3))

        if len(ids2) > len(ids3):
            not_verified_.append(group)
            continue

        idx = 0
        while not sorted_:
            if idx == len(ids1):
                verified_.append(group)
                break

            if ids1[idx] not in ids3:
                not_verified_.append(group)
                sorted_ = True
            idx += 1

    print(f'separated {len(groups_)} to {len(verified_)} verified and {len(not_verified_)} not verified')
    return verified_, not_verified_


def evaluate_matches(df, plot=False):
    """print report and plot to compare results to the pre labeled data"""
    print('evaluating matches')
    verified_ = []
    verified_false_ = []
    non_verified_ = []
    for i in tqdm(df.index):
        match_ = df.loc[i]
        hw_index = match_['index1']
        lda_index = match_['index2']
        if pd.notnull(ldu_raw.loc[lda_index, 'HW Account']):
            if ldu_raw.loc[lda_index, 'HW Account'] == hw_raw.loc[hw_index, 'account_id']:
                verified_.append(i)
            else:
                verified_false_.append(i)
        else:
            non_verified_.append(i)

    num_true_matches = len(ldu_raw.loc[pd.notnull(ldu_raw['HW Account'])])
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

    plt.savefig(f'plots/combined_plot.png', facecolor=face_color)
    plt.clf()
    print(f'saved plot: "combined_plot.png"')


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

    if save:
        df.to_pickle('data/generated/scores.pkl')
        print('saved new scores in: data/generated/scores.pkl')
    print('finished finale score calculation')

    return df


# make sure all datatypes are correct
hw = validate_strings(hw)
ldu = validate_strings(ldu)
hw_raw = validate_strings(hw_raw)
ldu_raw = validate_strings(ldu_raw)

scores = calc_combined_scores(scores, save=False)  # finale score through score formula


# scores_description = scores.describe()

# plot_combined(scores)


def match(df):
    """
    decide which comparisons are considered to be a match.

    using different conditions to define what is a match, a match must meet at least one of them.
    """
    masks = []

    # combined score is not too small and, name  or company_name matches and, email or phone matches
    masks.append((((df['name'] >= 0.8) | (df['company_name'] >= 0.9)) &
                  ((df['email'] >= 0.9) | (df['phone'] >= 0.9))))

    # combined score is not small and, all location attributes match
    masks.append((df['city'] >= 0.9) & (df['state'] >= 0.9) & (df['zip'] >= 1) &
                 (df['country'] >= 1) & (df['address'] >= 1))

    # combined score is not small and, some location attributes and the phone match
    masks.append((df['city'] >= 0.8) & (df['state'] >= 0.8) & (df['zip'] >= 0.8) &
                 (df['country'] >= 0.9) & (df['phone'] >= 1))

    # name or email or company_name is a perfect match
    masks.append((df['name'] == 1) | (df['company_name'] == 1) | (df['email'] == 1))

    print(f"\ncontact mask: {len(df.loc[masks[0]])} | "
          f"unique: {len(df.loc[masks[0] & ~ (masks[1] | masks[2] | masks[3])])}",
          f"\naddress mask: {len(df.loc[masks[1]])} | "
          f"unique: {len(df.loc[masks[1] & ~ (masks[0] | masks[2] | masks[3])])}",
          f"\nphone mask: {len(df.loc[masks[2]])} | "
          f"unique: {len(df.loc[masks[2] & ~ (masks[1] | masks[0] | masks[3])])}",
          f"\nexact mask: {len(df.loc[masks[3]])} | "
          f"unique: {len(df.loc[masks[3] & ~ (masks[0] | masks[1] | masks[2])])}")

    mask = masks[0]
    for m in masks:  # combine all matches
        mask = mask | m

    return df.loc[mask]


matches = match(scores)  # choose what is considered as a match
phone_matches = scores.loc[(scores['phone'] == 1) & scores['country'] == 1]
print(f'generated matches: {len(matches)}\n')

# verified, verified_false, non_verified = evaluate_matches(matches, plot=False)

groups, links = group_matches(matches)  # create groups from all matches

length_description = validate_groups(groups, print_dupes=False)

# separate all groups to need or not need verification
verified, not_verified = separate_groups(hw, ldu, groups)

# convert groups list to  dataframe
grouped_matches_all = groups_to_df(hw, ldu, groups, links)
grouped_matches_verified = groups_to_df(hw, ldu, verified, links)
grouped_matches_not_verified = groups_to_df(hw, ldu, not_verified, links)
grouped_matches_all_raw = groups_to_df(hw_raw, ldu_raw, groups, links)
grouped_matches_verified_raw = groups_to_df(hw_raw, ldu_raw, verified, links)
grouped_matches_not_verified_raw = groups_to_df(hw_raw, ldu_raw, not_verified, links)

# prepare dataframes to presentation
grouped_matches_all = to_presentation(grouped_matches_all, drop_hw_id=True)
grouped_matches_verified = to_presentation(grouped_matches_verified, drop_hw_id=True)
grouped_matches_not_verified = to_presentation(grouped_matches_not_verified, drop_hw_id=True)
grouped_matches_all_raw = to_presentation(grouped_matches_all_raw, drop_hw_id=True)
grouped_matches_verified_raw = to_presentation(grouped_matches_verified_raw, drop_hw_id=True)
grouped_matches_not_verified_raw = to_presentation(grouped_matches_not_verified_raw, drop_hw_id=True)

if __name__ == '__main__':
    with pd.ExcelWriter('data/generated/matches_ldu_hw.xlsx') as writer:  # save to excel
        grouped_matches_all_raw.to_excel(writer, sheet_name='all')
        grouped_matches_verified_raw.to_excel(writer, sheet_name='verified')
        grouped_matches_not_verified_raw.to_excel(writer, sheet_name='not verified')

print('saved results')

import pandas as pd
import recordlinkage as rl
import textdistance as td
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

hw = pd.read_pickle('data/generated/hw_processed.pkl')
lda = pd.read_pickle('data/generated/lda_processed.pkl')


# make sure datatypes are correct
def validate_strings(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else x)
        df[col] = df[col].apply(lambda x: x.removesuffix('.0') if pd.notnull(x) and x.endswith('.0') else x)
    return df


hw = validate_strings(hw)
lda = validate_strings(lda)

# hw = hw.loc[:]  # test on smaller dataset
# ldu = ldu.loc[:]  # test on smaller dataset

# compare specific column to fitting columns
hw_to_lda = {
    'email': ['email'],
    'company_name': ['name'],
    'name': ['name'],
    'name2': ['name'],
    'group': ['group'],
    'phone': ['phone', 'fax'],
    'address': ['address'],
    'address2': ['address'],
    'city': ['city'],
    'state': ['state'],
    'zip': ['zip'],
    'country': ['country'],
    'phone2': ['phone', 'fax'],
    'address3': ['address'],
    'address4': ['address'],
    'city2': ['city'],
    'state2': ['state'],
    'zip2': ['zip'],
    'country2': ['country'],
    'phone3': ['phone', 'fax']
}

# for column in keys use the max score of the comparisons from the fitting values
property_merge = {
    'name': ['name', 'name2'],
    'address': ['address', 'address2', 'address3', 'address4'],
    'city': ['city', 'city2'],
    'state': ['state', 'state2'],
    'zip': ['zip', 'zip2'],
    'country': ['country', 'country2'],
    'phone': ['phone', 'phone2', 'phone3']
}

# Levenshtein: smallest number of edits needed to transform one string to another
# ratcliff_obershelp: The score is twice the number of characters found in common divided by the total
# number of characters in the two strings

# compare values of certain type using certain algorithm
compare_method = {
    'email': td.ratcliff_obershelp,  # sequence based
    'name': td.ratcliff_obershelp,  # sequence based
    'address': td.ratcliff_obershelp,  # sequence based
    'city': td.levenshtein.normalized_similarity,  # edit distance based
    'state': td.levenshtein.normalized_similarity,  # edit distance based
    'zip': td.levenshtein.normalized_similarity,  # edit distance based
    'country': td.levenshtein.normalized_similarity,  # edit distance based
    'phone': td.levenshtein.normalized_similarity,  # edit distance based
    'group': td.levenshtein.normalized_similarity  # edit distance based
}


def compare_records(rec1, rec2, idx1, idx2):
    """
    compare two records/rows and return dict with similarity score between values.
    Compare rows of np array instead of pd series for better performance.
    """
    res = {}  # return ndict

    for col in idx1.keys():  # available columns from rec1
        if col not in set(hw_to_lda.keys()):  # ignore value if not in conversion dict
            continue
        temp_scores = []
        for col2 in hw_to_lda[col]:  # available columns from rec2

            # one of the values is missing
            if type(rec1[idx1[col]]) == float and not pd.notnull(rec1[idx1[col]]) or \
                    type(rec2[idx2[col2]]) == float and not pd.notnull(rec2[idx2[col2]]):
                temp_scores.append(0)

            # column in compare_method dict
            elif True in [key in col for key in compare_method.keys()]:
                for key in compare_method.keys():
                    if key in col:
                        temp_scores.append(compare_method[key](rec1[idx1[col]], rec2[idx2[col2]]))
                        continue

            else:
                temp_scores.append(-1)

        res[col] = float(max(temp_scores))

    for k, v in property_merge.items():
        res[k] = float(max([res[a] for a in v]))
        for a in v:
            if a != k:
                res.pop(a)

    return res


def worker(df1: pd.DataFrame, df2: pd.DataFrame, links: list, res_dict, n_proc: int):
    """
    function to run as a sub process to compare a slice of the possible links

    :return: a dataframe containing the indexes compared, and the similarity scores between values.
    """
    print(f'started process num {n_proc} with length {len(links)}')

    # create conversion dicts between index and column to simplify usage with np array
    indexer1 = dict(zip(list(df1.columns), list(range(df1.shape[1]))))
    indexer2 = dict(zip(list(df2.columns), list(range(df2.shape[1]))))

    res_scores = {}  # dict to be converted to df, for performance reasons

    # columns of the return df
    columns = ['index1', 'index2'] + list(compare_records(df1.values[0], df2.values[0], indexer1, indexer2).keys())
    for col in columns:
        res_scores[col] = []

    p_bar = tqdm(total=len(links))  # progressbar

    for pair in links:  # pair of indexes to compare

        temp = compare_records(df1.values[pair[0]], df2.values[pair[1]], indexer1, indexer2)
        temp['index1'] = pair[0]
        temp['index2'] = pair[1]

        for key, value in temp.items():  # attach results to new row
            res_scores[key].append(value)

        if n_proc == 0:  # update progressbar
            p_bar.update()

    res_df = pd.DataFrame(data=res_scores)  # convert results dict to df
    res_df['total'] = res_df.drop(columns=['index1', 'index2']).sum(axis=1)  # generate column with total score
    res_df = res_df.astype(float)
    res_dict[n_proc] = res_df  # return through shared dict between processes
    print(f'finished process num {n_proc}')


if __name__ == "__main__":

    indexer = rl.Index()  # indexer class to create list of all possible links between dataframes
    indexer.full()
    possible_links = indexer.index(hw, lda)  # create list of all possible links
    print(f'total possible links: {len(possible_links)}')

    manager = mp.Manager()
    return_dict = manager.dict()  # shared memory dict to retrieve results from sub processes
    proc_num = mp.cpu_count() - 2  # use all but 2 cores
    link_splits = np.array_split(possible_links, proc_num)  # split links list to equal parts

    jobs = []
    for i in range(proc_num):
        p = mp.Process(target=worker, args=(hw, lda, link_splits[i], return_dict, i))  # create process
        jobs.append(p)
        p.start()  # start process

    for proc in jobs:
        proc.join()  # wait for process to finish

    frames = [return_dict[i] for i in range(proc_num)]  # list of all dataframes in correct order
    scores = pd.concat(frames, ignore_index=True)  # merge dataframes
    scores.to_pickle('data/generated/scores_ldu_hw.pkl')  # save results
    print('finished all operations')

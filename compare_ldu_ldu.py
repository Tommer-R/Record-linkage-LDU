import pandas as pd
import recordlinkage as rl
import textdistance as td
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

lda = pd.read_pickle('data/processed/lda_processed.pkl')


# make sure datatypes are correct
def validate_strings(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else x)
        df[col] = df[col].apply(lambda x: x.removesuffix('.0') if pd.notnull(x) and x.endswith('.0') else x)
    return df


lda = validate_strings(lda)  # make sure datatypes are correct

# ldu = ldu.loc[:100]  # test on smaller dataset

# Levenshtein: smallest number of edits needed to transform one string to another
# ratcliff_obershelp: The score is twice the number of characters found in common divided by the total
# number of characters in the two strings

# compare values of certain type using certain algorithm
compare_method = {
    'email': td.ratcliff_obershelp,  # sequence based
    'web_site': td.ratcliff_obershelp,  # sequence based
    'name': td.ratcliff_obershelp,  # sequence based
    'address': td.ratcliff_obershelp,  # sequence based
    'city': td.levenshtein.normalized_similarity,  # edit distance based
    'state': td.levenshtein.normalized_similarity,  # edit distance based
    'zip': td.levenshtein.normalized_similarity,  # edit distance based
    'country': td.levenshtein.normalized_similarity,  # edit distance based
    'phone': td.levenshtein.normalized_similarity,  # edit distance based
    'fax': td.levenshtein.normalized_similarity,  # edit distance based
    'group': td.levenshtein.normalized_similarity  # edit distance based
}


def compare_records(rec1, rec2, idx1):
    """
    compare two records/rows and return dict with similarity score between values.
    Compare rows of np array instead of pd series for better performance.
    """
    res = {}  # return ndict

    for col in idx1.keys():  # available columns
        temp_score = 0

        # one of the values is missing
        if type(rec1[idx1[col]]) == float and not pd.notnull(rec1[idx1[col]]) or \
                type(rec2[idx1[col]]) == float and not pd.notnull(rec2[idx1[col]]):
            temp_score = 0

        # column in compare_method dict
        elif col in compare_method.keys():
            for key in compare_method.keys():
                if key in col:
                    temp_score = compare_method[key](rec1[idx1[col]], rec2[idx1[col]])
                    continue

        res[col] = float(temp_score)

    return res


def worker(df1: pd.DataFrame, links: list, res_dict, n_proc: int):
    """
    function to run as a sub process to compare a slice of the possible links

    :return: a dataframe containing the indexes compared, and the similarity scores between values.
    """
    print(f'started process num {n_proc} with length {len(links)}')

    # create conversion dict between index and column to simplify usage with np array
    indexer1 = dict(zip(list(df1.columns), list(range(df1.shape[1]))))

    res_scores = {}  # dict to be converted to df, for performance reasons

    # columns of the return df
    columns = ['index1', 'index2'] + list(compare_records(df1.values[0], df1.values[0], indexer1).keys())
    for col in columns:
        res_scores[col] = []

    p_bar = tqdm(total=len(links))  # progressbar

    for pair in links:  # pair of indexes to compare

        temp = compare_records(df1.values[pair[0]], df1.values[pair[1]], indexer1)
        temp['index1'] = pair[0]
        temp['index2'] = pair[1]

        for key, value in temp.items():  # attach results to new row
            res_scores[key].append(value)

        if n_proc == 0:  # update progressbar
            p_bar.update()

    res_df = pd.DataFrame(data=res_scores)  # convert results dict to df
    # generate column with total score
    res_df['total'] = res_df.drop(columns=['index1', 'index2', 'id', 'hw id']).sum(axis=1)
    res_df = res_df.astype(float)
    res_dict[n_proc] = res_df  # return through shared dict between processes
    print(f'finished process num {n_proc}')


if __name__ == "__main__":

    indexer = rl.Index()  # indexer class to create list of all possible links between dataframes
    indexer.full()
    possible_links = indexer.index(lda)  # create list of all possible links
    print(f'total possible links: {len(possible_links)}')

    manager = mp.Manager()
    return_dict = manager.dict()  # shared memory dict to retrieve results from sub processes
    proc_num = mp.cpu_count() - 3  # use all but 2 cores
    link_splits = np.array_split(possible_links, proc_num)  # split links list to equal parts

    jobs = []
    for i in range(proc_num):
        p = mp.Process(target=worker, args=(lda, link_splits[i], return_dict, i))  # create process
        jobs.append(p)
        p.start()  # start process

    for proc in jobs:
        proc.join()  # wait for process to finish

    frames = [return_dict[i] for i in range(proc_num)]  # list of all dataframes in correct order
    scores = pd.concat(frames, ignore_index=True)  # merge dataframes
    scores.drop(columns=['id', 'hw id'], inplace=True)
    scores.to_pickle('data/generated/scores_ldu_ldu.pkl')  # save results
    print('finished all operations')

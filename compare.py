import pandas as pd
import recordlinkage as rl
import textdistance as td
from time import time
import multiprocessing as mp
import numpy as np
from datetime import timedelta, datetime
from tqdm import tqdm

hw = pd.read_pickle('data/processed/hw_processed.pkl')
lda = pd.read_pickle('data/processed/lda_processed.pkl')


def validate_strings(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else x)
        df[col] = df[col].apply(lambda x: x.removesuffix('.0') if pd.notnull(x) and x.endswith('.0') else x)
    return df


hw = validate_strings(hw)
lda = validate_strings(lda)

hw = hw.loc[:]
lda = lda.loc[:]

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

property_merge = {
    'name': ['name', 'name2'],
    'address': ['address', 'address2', 'address3', 'address4'],
    'city': ['city', 'city2'],
    'state': ['state', 'state2'],
    'zip': ['zip', 'zip2'],
    'country': ['country', 'country2'],
    'phone': ['phone', 'phone2', 'phone3']
}

compare_method = {
    'email': td.ratcliff_obershelp,
    'name': td.ratcliff_obershelp,
    'address': td.ratcliff_obershelp,
    'city': td.levenshtein.normalized_similarity,
    'state': td.levenshtein.normalized_similarity,
    'zip': td.levenshtein.normalized_similarity,
    'country': td.levenshtein.normalized_similarity,
    'phone': td.levenshtein.normalized_similarity,
    'group': td.levenshtein.normalized_similarity
}


def compare_records(rec1, rec2, idx1, idx2):
    res = {}

    for col in idx1.keys():
        if col not in set(hw_to_lda.keys()):
            continue
        temp_scores = []
        for col2 in hw_to_lda[col]:
            if type(rec1[idx1[col]]) == float and not pd.notnull(rec1[idx1[col]]) or \
                    type(rec2[idx2[col2]]) == float and not pd.notnull(rec2[idx2[col2]]):
                temp_scores.append(0)

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


def worker(df1, df2, links, res_dict, n_proc):
    print(f'started process num {n_proc} with length {len(links)}')
    s_time = time()
    last_percent = 0.0

    indexer1 = dict(zip(list(df1.columns), list(range(df1.shape[1]))))
    indexer2 = dict(zip(list(df2.columns), list(range(df2.shape[1]))))

    res_scores = {}
    columns = ['index1', 'index2'] + list(compare_records(df1.values[0], df2.values[0], indexer1, indexer2).keys())
    for col in columns:
        res_scores[col] = []

    counter = 0
    # if n_proc == 0:
    # p_bar = tqdm(total=len(links))

    for pair in tqdm(links):

        temp = compare_records(df1.values[pair[0]], df2.values[pair[1]], indexer1, indexer2)
        temp['index1'] = pair[0]
        temp['index2'] = pair[1]

        for key, value in temp.items():
            res_scores[key].append(value)

        counter += 1

    res_df = pd.DataFrame(data=res_scores)
    res_df['total'] = res_df.drop(columns=['index1', 'index2']).sum(axis=1)
    res_df = res_df.astype(float)
    res_dict[n_proc] = res_df
    print(f'finished process num {n_proc}')


if __name__ == "__main__":

    indexer = rl.Index()
    indexer.full()
    possible_links = indexer.index(hw, lda)

    manager = mp.Manager()
    return_dict = manager.dict()
    proc_num = mp.cpu_count() - 2
    link_splits = np.array_split(possible_links, proc_num)

    jobs = []
    for i in range(proc_num):
        p = mp.Process(target=worker, args=(hw, lda, link_splits[i], return_dict, i))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    frames = [return_dict[i] for i in range(proc_num)]
    scores = pd.concat(frames, ignore_index=True)
    scores.to_pickle('data/generated/scores.pkl')
    print('finished all operations')

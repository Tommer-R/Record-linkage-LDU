import pandas as pd
import recordlinkage as rl
import textdistance as td
from time import time
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

lda = pd.read_pickle('data/processed/lda_processed.pkl')


def validate_strings(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else x)
        df[col] = df[col].apply(lambda x: x.removesuffix('.0') if pd.notnull(x) and x.endswith('.0') else x)
    return df


lda = validate_strings(lda)

lda = lda.loc[:]


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
    'web_site': td.ratcliff_obershelp,
    'name': td.ratcliff_obershelp,
    'address': td.ratcliff_obershelp,
    'city': td.levenshtein.normalized_similarity,
    'state': td.levenshtein.normalized_similarity,
    'zip': td.levenshtein.normalized_similarity,
    'country': td.levenshtein.normalized_similarity,
    'phone': td.levenshtein.normalized_similarity,
    'fax': td.levenshtein.normalized_similarity,
    'group': td.levenshtein.normalized_similarity
}


def compare_records(rec1, rec2, idx1):
    res = {}
    for col in idx1.keys():
        temp_score = 0
        if type(rec1[idx1[col]]) == float and not pd.notnull(rec1[idx1[col]]) or \
                type(rec2[idx1[col]]) == float and not pd.notnull(rec2[idx1[col]]):
            temp_score = 0

        elif col in compare_method.keys():
            for key in compare_method.keys():
                if key in col:
                    temp_score = compare_method[key](rec1[idx1[col]], rec2[idx1[col]])
                    continue

        res[col] = float(temp_score)

    return res


def worker(df1, links, res_dict, n_proc):
    print(f'started process num {n_proc} with length {len(links)}')

    indexer1 = dict(zip(list(df1.columns), list(range(df1.shape[1]))))
    res_scores = {}
    columns = ['index1', 'index2'] + list(compare_records(df1.values[0], df1.values[0], indexer1).keys())
    for col in columns:
        res_scores[col] = []

    counter = 0
    # if n_proc == 0:
    p_bar = tqdm(total=len(links))

    for pair in links:

        temp = compare_records(df1.values[pair[0]], df1.values[pair[1]], indexer1)
        temp['index1'] = pair[0]
        temp['index2'] = pair[1]

        for key, value in temp.items():
            res_scores[key].append(value)

        counter += 1
        if n_proc == 0:
            p_bar.update()

    res_df = pd.DataFrame(data=res_scores)
    res_df['total'] = res_df.drop(columns=['index1', 'index2', 'id', 'hw id']).sum(axis=1)
    res_df = res_df.astype(float)
    res_dict[n_proc] = res_df
    print(f'finished process num {n_proc}')


if __name__ == "__main__":

    indexer = rl.Index()
    indexer.full()
    possible_links = indexer.index(lda)
    print(f'total possible links: {len(possible_links)}')

    manager = mp.Manager()
    return_dict = manager.dict()
    proc_num = mp.cpu_count() - 2
    # proc_num = 1
    link_splits = np.array_split(possible_links, proc_num)

    jobs = []
    for i in range(proc_num):
        p = mp.Process(target=worker, args=(lda, link_splits[i], return_dict, i))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    frames = [return_dict[i] for i in range(proc_num)]
    scores = pd.concat(frames, ignore_index=True)
    scores.drop(columns=['id', 'hw id'], inplace=True)
    scores.to_pickle('data/generated/scores_ldu_ldu.pkl')
    print('finished all operations')

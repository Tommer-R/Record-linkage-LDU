from pre_processing import hw, lda
import pandas as pd
import recordlinkage as rl
import textdistance as td
from progressbar import progressbar
import warnings
import multiprocessing as mp
import numpy as np


hw_to_lda = {
    'email': ['email'],
    'company_name': ['name'],
    'name1': ['name'],
    'name2': ['name'],
    'group': ['group'],
    'phone': ['phone', 'fax'],
    'address1': ['address'],
    'address2': ['address'],
    'city': ['city'],
    'state': ['state code', 'state'],
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
    'name': ['name1', 'name2'],
    'address': ['address1', 'address2', 'address3', 'address4'],
    'city': ['city', 'city2'],
    'state': ['state', 'state2'],
    'zip': ['zip', 'zip2'],
    'country': ['country', 'country2'],
    'phone': ['phone', 'phone2', 'phone3']
}

compare_method = {
    'tokens': td.jaccard,
    'string': td.levenshtein.normalized_similarity
}


def compare_records(rec1, rec2):
    scores = {}

    for col in rec1.index:
        if col not in set(hw_to_lda.keys()):
            continue
        temp_scores = []
        for col2 in hw_to_lda[col]:
            if type(rec1[col]) == float and not pd.notnull(rec1[col]) or \
                    type(rec2[col2]) == float and not pd.notnull(rec2[col2]):
                temp_scores.append(0)
            elif type(rec1[col]) == list and type(rec2[col2]) == list:
                temp_scores.append(compare_method['tokens'](rec1[col], rec2[col2]))

            elif type(rec1[col]) == str and type(rec2[col2]) == str:
                temp_scores.append(compare_method['string'](rec1[col], rec2[col2]))

            else:
                warnings.warn(f'wrong types: {rec1[col]} = {type(rec1[col])} | {rec2[col2]} = {type(rec2[col2])}')

        scores[col] = float(max(temp_scores))

    for k, v in property_merge.items():
        scores[k] = float(max([scores[a] for a in v]))
        for a in v:
            if a != k:
                scores.pop(a)

    return scores


def worker(df1, df2, links, return_dict, proc):
    scores = pd.DataFrame(index=possible_links, columns=list(compare_records(hw.loc[1], lda.loc[1]).keys()))
    for pair in progressbar(possible_links):
        scores.loc[pair] = compare_records(hw.loc[pair[0]], lda.loc[pair[1]])

    scores['total'] = scores.sum(axis=1)

    scores = scores.astype(float)


if __name__ == "__main__":

    # lda.info()
    # hw.info()

    indexer = rl.Index()
    indexer.full()
    possible_links = indexer.index(hw, lda)

    manager = mp.Manager()
    return_dict = manager.dict()
    proc_num = 5
    link_splits = np.array_split(possible_links, proc_num)

    jobs = []
    for i in range(proc_num):
        p = mp.Process(target=worker, args=(hw, lda, link_splits[i], return_dict, i))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(return_dict.values())

#     description = scores.describe()


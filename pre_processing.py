import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from itertools import product
import us
import country_converter as coco

# download nltk resources on first run
# nltk.download('stopwords')
# nltk.download('punkt')

all_stopwords = stopwords.words('english')
all_stopwords.append('&')
state_codes = ['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga', 'hi', 'ia', 'id', 'il', 'in', 'ks',
               'ky', 'la', 'ma', 'md', 'me', 'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm', 'nv',
               'ny', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy']

# make sure state codes are not removed
for s in state_codes:
    try:
        all_stopwords.remove(s)
    except ValueError:
        pass


address_stopwords = all_stopwords + ["street", "st", "place", "rd", "road", 'square', 'ave']
name_stopwords = all_stopwords + ['co', 'corp', 'inc', 'company', 'limited', 'llc']


def fix_state(x: str):
    a = us.states.lookup(x)
    if a is not None:
        return a.name.lower()
    else:
        return x


def fix_country(x: str):
    a = coco.convert(x, to='name_short')
    if a != 'not found':
        return a.lower()
    else:
        return x


def normalize_name(a: str):
    a = a.lower()  # convert to lower case
    a = re.sub(r'[^a-z0-9 ]', '', a)  # keep numbers and letters and space
    tokens = word_tokenize(a)  # separate words
    tokens = [word for word in tokens if word not in name_stopwords]  # remove stopwords
    tokens.sort()
    res = ' '.join(tokens)
    return res


def normalize_address(a: str):
    a = a.lower()  # convert to lower case
    a = re.sub(r'[^a-z0-9 ]', '', a)  # keep numbers and letters and space
    tokens = word_tokenize(a)  # separate words
    tokens = [word for word in tokens if word not in address_stopwords]  # remove stopwords
    tokens.sort()
    res = ' '.join(tokens)
    return res


def normalize_number(a) -> str:
    a = re.sub(r'[^0-9]', '', a)  # keep only numbers
    return a


def normalize_email(a):
    a = a.lower()  # convert to lower case
    a = re.sub(r'[^a-z0-9 ]', ' ', a)  # keep numbers and letters and space
    tokens = word_tokenize(a)  # separate words
    tokens = [word for word in tokens if word not in all_stopwords]  # remove stopwords
    res = ' '.join(tokens)
    return res


def remove_www(a):
    if type(a) == list and 'www' in a:
        a.remove('www')
    return a


def merge_columns(df, col1, col2, drop):
    temp_df = pd.DataFrame(index=list(df.index), columns=[col1])
    for idx in df.index:
        if type(df.loc[idx, col1]) == list and type(df.loc[idx, col2]) == list:
            df.loc[idx, col1].extend(df.loc[idx, col2])
            temp_df.loc[idx, col1] = df.loc[idx, col1]
        elif type(df.loc[idx, col1]) == str and type(df.loc[idx, col2]) == str:
            temp_df.loc[idx, col1] = df.loc[idx, col1] + ' ' + df.loc[idx, col2]
        elif type(df.loc[idx, col1]) == str and type(df.loc[idx, col2]) == list:
            df.loc[idx, col2].append(df.loc[idx, col1])
            temp_df.loc[idx, col1] = df.loc[idx, col2]
        elif type(df.loc[idx, col1]) == list and type(df.loc[idx, col2]) == str:
            df.loc[idx, col1].append(df.loc[idx, col2])
            temp_df.loc[idx, col1] = df.loc[idx, col1]
        elif type(df.loc[idx, col1]) == float and not pd.notnull(df.loc[idx, col1]) and type(df.loc[idx, col2]) == list:
            temp_df.loc[idx, col1] = df.loc[idx, col2]
        else:
            temp_df.loc[idx, col1] = df.loc[idx, col1]

    df[col1] = temp_df[col1]

    if drop:
        df.drop(columns=[col2], inplace=True)
    return df


ldu = pd.read_csv('data/raw/Priority Customers.csv', delimiter=';')
hw = pd.read_csv('data/raw/HeroWeb Accounts.csv', delimiter=';')
print('imported data')


ldu.columns = [c.lower() for c in list(ldu.columns)]  # change to lower case
hw.columns = [c.lower() for c in list(hw.columns)]  # change to lower case
hw.columns = [c.replace('account_', '') for c in list(hw.columns)]  # simplify x names
hw_columns_to_drop = ['active', 'date_joined', 'date_expires', 'referred_by', 'locked', 'terms', 'sales_rep',
                      'is_sales_rep', 'tax_id', 'tax_exempt', 'long', 'lat', 'date_last_ordered', 'total_orders',
                      'total_revenue', 'notes', 'store_optin']
ldu_columns_to_drop = ['city & state', 'state code']
hw = hw[~(hw['active'] == 'N')]  # drop inactive records
hw.drop(columns=hw_columns_to_drop, inplace=True)
ldu.drop(columns=ldu_columns_to_drop, inplace=True)

print('dropped and refactored columns')


# rename columns to match each other
hw.columns = ['id', 'email', 'company_name', 'last_name', 'first_name', 'name2', 'group', 'phone','address', 'address2',
              'city', 'state', 'zip', 'country', 'phone2', 'saddress1', 'saddress2', 'city2', 'state2', 'zip2',
              'country2', 'phone3']

ldu.columns = ['id', 'name', 'phone', 'fax', 'email', 'group', 'address1', 'address2', 'address3', 'city', 'state',
               'zip', 'country', 'web_site', 'hw id']
print('renamed columns')

# drop irrelevant rows
for index, row in ldu.iterrows():
    try:
        int(row['id'])
    except ValueError:
        ldu.drop(index, inplace=True)

inactive_ldu = ['100014', '100060', '100109', '100110', '100111', '100112', '100113', '100632', '101084', '101085',
                '101124', '101297', '101451', '101453', '101454', '101951', '102101', '103174', '103325', '103326',
                '103424', '103767', '104112', '104113', '104315', '104458', '104468', '105075', '105214']
ldu = ldu[~ldu['id'].isin(inactive_ldu)]  # drop inactive records
hw.reset_index(inplace=True)
ldu.reset_index(inplace=True)
print('dropped irrelevant rows')

# fix state names
hw['state'] = hw['state'].apply(lambda x: fix_state(x) if pd.notnull(x) else x)
hw['state2'] = hw['state2'].apply(lambda x: fix_state(x) if pd.notnull(x) else x)
ldu['state'] = ldu['state'].apply(lambda x: fix_state(x) if pd.notnull(x) else x)
print('unified states format')

# fix country names
hw['country'] = hw['country'].apply(lambda x: 'usa' if pd.notnull(x) and x.lower() == 'un' else x)
hw['country'] = hw['country'].apply(lambda x: fix_country(x) if pd.notnull(x) else x)
hw['country2'] = hw['country2'].apply(lambda x: 'usa' if pd.notnull(x) and x.lower() == 'un' else x)
hw['country2'] = hw['country2'].apply(lambda x: fix_country(x) if pd.notnull(x) else x)
ldu['country'] = ldu['country'].apply(lambda x: fix_country(x) if pd.notnull(x) else x)
print('unified countries format')

# separate to raw and processed datasets
lda_raw = ldu.copy()
hw_raw = hw.copy()

# normalize ldu values using the fitting functions
ldu['name'] = ldu['name'].apply(lambda x: normalize_name(x) if pd.notnull(x) else x)
ldu['phone'] = ldu['phone'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
ldu['fax'] = ldu['fax'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
ldu['email'] = ldu['email'].apply(lambda x: normalize_email(x) if pd.notnull(x) else x)
ldu['group'] = ldu['group'].apply(lambda x: normalize_name(x) if pd.notnull(x) else x)
ldu['address1'] = ldu['address1'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
ldu['address2'] = ldu['address2'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
ldu['address3'] = ldu['address3'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
ldu['city'] = ldu['city'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
ldu['state'] = ldu['state'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
ldu['zip'] = ldu['zip'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
ldu['country'] = ldu['country'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
ldu['web_site'] = ldu['web_site'].apply(lambda x: normalize_email(x) if pd.notnull(x) else x)
ldu['web_site'] = ldu['web_site'].apply(lambda x: remove_www(x) if x != np.nan else x)  # remove 'www'

# normalize hw values using the fitting functions
hw['email'] = hw['email'].apply(lambda x: normalize_email(x) if pd.notnull(x) else x)
hw['company_name'] = hw['company_name'].apply(lambda x: normalize_name(x) if pd.notnull(x) else x)
hw['last_name'] = hw['last_name'].apply(lambda x: normalize_name(x) if pd.notnull(x) else x)
hw['first_name'] = hw['first_name'].apply(lambda x: normalize_name(x) if pd.notnull(x) else x)
hw['name2'] = hw['name2'].apply(lambda x: normalize_name(x) if pd.notnull(x) else x)
hw['group'] = hw['group'].apply(lambda x: normalize_name(x) if pd.notnull(x) else x)
hw['phone'] = hw['phone'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
hw['address'] = hw['address'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['address2'] = hw['address2'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['city'] = hw['city'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['state'] = hw['state'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['zip'] = hw['zip'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
hw['country'] = hw['country'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['phone2'] = hw['phone2'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
hw['saddress1'] = hw['saddress1'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['saddress2'] = hw['saddress2'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['city2'] = hw['city2'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['state2'] = hw['state2'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['zip2'] = hw['zip2'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
hw['country2'] = hw['country2'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['phone3'] = hw['phone3'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
print('normalized columns')

# convert to string
for col, i in product(list(hw.columns), hw.index):
    if type(hw[col][i]) == float and pd.notnull(hw[col][i]):
        hw.loc[i, col] = str(hw.loc[i, col])

# merge names columns and rename the column
hw = merge_columns(hw, 'first_name', 'last_name', drop=True)
hw.rename({'first_name': 'name', 'saddress1': 'address3', 'saddress2': 'address4'}, axis=1, inplace=True)
hw['name'] = hw['name'].apply(lambda x: normalize_name(x) if pd.notnull(x) else x)


# merge names columns and rename the column
hw_raw = merge_columns(hw_raw, 'first_name', 'last_name', drop=True)
hw_raw.rename({'first_name': 'name', 'saddress1': 'address3', 'saddress2': 'address4'}, axis=1, inplace=True)

# remove duplicate values within a record
for i in hw.index:
    if hw.loc[i, 'address'] == hw.loc[i, 'address2']:
        hw.loc[i, 'address2'] = np.nan

    if hw.loc[i, 'city'] == hw.loc[i, 'city2'] and type(hw.loc[i, 'address2']) == float and \
            not pd.notnull(hw.loc[i, 'address2']):
        hw.loc[i, 'city2'] = np.nan

    if hw.loc[i, 'state'] == hw.loc[i, 'state2'] and type(hw.loc[i, 'address2']) == float and \
            not pd.notnull(hw.loc[i, 'address2']):
        hw.loc[i, 'state2'] = np.nan

    if hw.loc[i, 'zip'] == hw.loc[i, 'zip2'] and type(hw.loc[i, 'address2']) == float and \
            not pd.notnull(hw.loc[i, 'address2']):
        hw.loc[i, 'zip2'] = np.nan

    if hw.loc[i, 'country'] == hw.loc[i, 'country2'] and type(hw.loc[i, 'address2']) == float and \
            not pd.notnull(hw.loc[i, 'address2']):
        hw.loc[i, 'country2'] = np.nan

    if hw.loc[i, 'phone2'] == hw.loc[i, 'phone3'] or hw.loc[i, 'phone'] == hw.loc[i, 'phone3'] and \
            type(hw.loc[i, 'address2']) == float and not pd.notnull(hw.loc[i, 'address2']):
        hw.loc[i, 'phone3'] = np.nan

    if hw.loc[i, 'phone'] == hw.loc[i, 'phone2'] and type(hw.loc[i, 'address2']) == float and \
            not pd.notnull(hw.loc[i, 'address2']):
        hw.loc[i, 'phone2'] = np.nan

ldu = merge_columns(ldu, 'address1', 'address2', drop=True)
ldu = merge_columns(ldu, 'address1', 'address3', drop=True)

ldu = ldu.rename({'address1': 'address'}, axis=1)

lda_raw = merge_columns(lda_raw, 'address1', 'address2', drop=True)
lda_raw = merge_columns(lda_raw, 'address1', 'address3', drop=True)

lda_raw = lda_raw.rename({'address1': 'address'}, axis=1)
print('merged columns')

# save results
hw.to_pickle('data/generated/hw_processed.pkl')
ldu.to_pickle('data/generated/ldu_processed.pkl')

hw_raw.to_pickle('data/raw/hw_raw.pkl')
lda_raw.to_pickle('data/raw/ldu_raw.pkl')

print('finished operations')

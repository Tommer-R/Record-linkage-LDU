import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from itertools import product
import us
import country_converter as coco


# nltk.download('stopwords')
# nltk.download('punkt')

all_stopwords = stopwords.words('english')
all_stopwords.append('&')
state_codes = ['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga', 'hi', 'ia', 'id', 'il', 'in', 'ks',
               'ky', 'la', 'ma', 'md', 'me', 'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm', 'nv',
               'ny', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy']

for s in state_codes:
    try:
        all_stopwords.remove(s)
    except ValueError:
        pass


address_stopwords = all_stopwords + ["street", "st", "place", "rd", "road", 'square']
name_stopwords = all_stopwords + ['co', 'corp', 'inc', 'company', 'limited', 'llc']

lda = pd.read_csv('data/raw/Priority Customers.csv', delimiter=';')
hw = pd.read_csv('data/raw/HeroWeb Accounts.csv', delimiter=';')
print('imported data')
# lda = lda[:500]
# hw = hw[:500]

lda.columns = [c.lower() for c in list(lda.columns)]  # change to lower case
hw.columns = [c.lower() for c in list(hw.columns)]  # change to lower case
hw.columns = [c.replace('account_', '') for c in list(hw.columns)]  # simplify x names
hw_columns_to_drop = ['active', 'date_joined', 'date_expires', 'referred_by', 'locked', 'terms', 'sales_rep',
                      'is_sales_rep', 'tax_id', 'tax_exempt', 'long', 'lat', 'date_last_ordered', 'total_orders',
                      'total_revenue', 'notes', 'store_optin']
lda_columns_to_drop = ['city & state', 'state code']

hw.drop(columns=hw_columns_to_drop, inplace=True)
lda.drop(columns=lda_columns_to_drop, inplace=True)
print('dropped columns')


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


# rename columns to match
"""
HeroWeb

email_address > email
name > name2
telephone > phone
stelephone >
saddress1 > 
saddress2 > 
scity > 
sstate > 
szip > 
scountry > 
telephone2 > phone2
"""
hw.columns = ['id', 'email', 'company_name', 'last_name', 'first_name', 'name2', 'group', 'phone',
              'address', 'address2', 'city', 'state', 'zip', 'country', 'phone2', 'saddress1',
              'saddress2', 'city2', 'state2', 'zip2', 'country2', 'phone3']

lda.columns = ['id', 'name', 'phone', 'fax', 'email', 'group', 'address1', 'address2', 'address3', 'city',
               'state', 'zip', 'country', 'web_site', 'hw id']
print('renamed columns')


for index, row in lda.iterrows():
    try:
        int(row['id'])
    except ValueError:
        lda.drop(index, inplace=True)

hw['state'] = hw['state'].apply(lambda x: fix_state(x) if pd.notnull(x) else x)  # fix state names
hw['state2'] = hw['state2'].apply(lambda x: fix_state(x) if pd.notnull(x) else x)  # fix state names
lda['state'] = lda['state'].apply(lambda x: fix_state(x) if pd.notnull(x) else x)  # fix state names
print('unified states')

hw['country'] = hw['country'].apply(lambda x: 'usa' if pd.notnull(x) and x.lower() == 'un' else x)  # fix country names
hw['country'] = hw['country'].apply(lambda x: fix_country(x) if pd.notnull(x) else x)  # fix country names
hw['country2'] = hw['country2'].apply(lambda x: 'usa' if pd.notnull(x) and x.lower() == 'un' else x)  # fix country names
hw['country2'] = hw['country2'].apply(lambda x: fix_country(x) if pd.notnull(x) else x)  # fix country names
lda['country'] = lda['country'].apply(lambda x: fix_country(x) if pd.notnull(x) else x)  # fix country names
print('unified countries')


def normalize_name(a: str):
    a = a.lower()  # convert to lower case
    a = re.sub(r'[^a-z0-9 ]', '', a)  # keep numbers and letters and space
    tokens = word_tokenize(a)  # separate words
    res = ' '.join(tokens)
    return res


def normalize_address(a: str):
    a = a.lower()  # convert to lower case
    a = re.sub(r'[^a-z0-9 ]', '', a)  # keep numbers and letters and space
    tokens = word_tokenize(a)  # separate words
    tokens = [word for word in tokens if word not in address_stopwords]  # remove stopwords
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


def merge_columns(df, col1, col2, drop=True):
    temp_df = pd.DataFrame(index=list(df.index), columns=[col1])
    for i in df.index:
        if type(df.loc[i, col1]) == list and type(df.loc[i, col2]) == list:
            df.loc[i, col1].extend(df.loc[i, col2])
            temp_df.loc[i, col1] = df.loc[i, col1]
        elif type(df.loc[i, col1]) == str and type(df.loc[i, col2]) == str:
            temp_df.loc[i, col1] = df.loc[i, col1] + ' ' + df.loc[i, col2]
        elif type(df.loc[i, col1]) == str and type(df.loc[i, col2]) == list:
            df.loc[i, col2].append(df.loc[i, col1])
            temp_df.loc[i, col1] = df.loc[i, col2]
        elif type(df.loc[i, col1]) == list and type(df.loc[i, col2]) == str:
            df.loc[i, col1].append(df.loc[i, col2])
            temp_df.loc[i, col1] = df.loc[i, col1]
        elif type(df.loc[i, col1]) == float and not pd.notnull(df.loc[i, col1]) and type(df.loc[i, col2]) == list:
            temp_df.loc[i, col1] = df.loc[i, col2]
        else:
            temp_df.loc[i, col1] = df.loc[i, col1]

    df[col1] = temp_df[col1]

    if drop:
        df.drop(columns=[col2], inplace=True)
    return df


lda_raw = lda.copy()
hw_raw = hw.copy()

# normalize lda
lda['name'] = lda['name'].apply(lambda x: normalize_name(x) if pd.notnull(x) else x)
lda['phone'] = lda['phone'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
lda['fax'] = lda['fax'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
lda['email'] = lda['email'].apply(lambda x: normalize_email(x) if pd.notnull(x) else x)
lda['group'] = lda['group'].apply(lambda x: normalize_name(x) if pd.notnull(x) else x)
lda['address1'] = lda['address1'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
lda['address2'] = lda['address2'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
lda['address3'] = lda['address3'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
lda['city'] = lda['city'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
lda['state'] = lda['state'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
lda['zip'] = lda['zip'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
lda['country'] = lda['country'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
lda['web_site'] = lda['web_site'].apply(lambda x: normalize_email(x) if pd.notnull(x) else x)
lda['web_site'] = lda['web_site'].apply(lambda x: remove_www(x) if x != np.nan else x)  # remove 'www'

# normalize hw
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

hw = merge_columns(hw, 'first_name', 'last_name')
hw.rename({'first_name': 'name', 'saddress1': 'address3', 'saddress2': 'address4'}, axis=1, inplace=True)

hw_raw = merge_columns(hw_raw, 'first_name', 'last_name')
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

lda = merge_columns(lda, 'address1', 'address2')
lda = merge_columns(lda, 'address1', 'address3')

lda = lda.rename({'address1': 'address'}, axis=1)

lda_raw = merge_columns(lda_raw, 'address1', 'address2')
lda_raw = merge_columns(lda_raw, 'address1', 'address3')

lda_raw = lda_raw.rename({'address1': 'address'}, axis=1)
print('merged columns')

hw.replace([], np.nan, inplace=True)
lda.replace([], np.nan, inplace=True)
hw_raw.replace([], np.nan, inplace=True)
lda_raw.replace([], np.nan, inplace=True)

hw.to_pickle('data/processed/hw_processed.pkl')
lda.to_pickle('data/processed/lda_processed.pkl')

hw_raw.to_pickle('data/raw/hw_raw.pkl')
lda_raw.to_pickle('data/raw/lda_raw.pkl')

print('finished operations')



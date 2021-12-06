import pandas as pd
import numpy as np
import recordlinkage as rl
import textdistance as td
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

all_stopwords = stopwords.words('english')
all_stopwords.append('&')
address_stopwords = all_stopwords + ["street", "st", "place", "rd", "road", 'square']
name_stopwords = all_stopwords + ['co', 'corp', 'inc', 'company', 'limited', 'llc']

lda = pd.read_csv('Priority Customers.csv', delimiter=';')
hw = pd.read_csv('HeroWeb Accounts.csv', delimiter=';')

lda.columns = [c.lower() for c in list(lda.columns)]  # change to lower case
hw.columns = [c.lower() for c in list(hw.columns)]  # change to lower case
hw.columns = [c.replace('account_', '') for c in list(hw.columns)]  # simplify column names
hw_columns_to_drop = ['active', 'date_joined', 'date_expires', 'referred_by', 'locked', 'terms', 'sales_rep',
                      'is_sales_rep', 'tax_id', 'tax_exempt', 'long', 'lat', 'date_last_ordered', 'total_orders',
                      'total_revenue', 'notes', 'store_optin']
lda_columns_to_drop = ['city & state']

hw.drop(columns=hw_columns_to_drop, inplace=True)
lda.drop(columns=lda_columns_to_drop, inplace=True)

# rename columns to match
"""
HeroWeb

email_address > email
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
hw.columns = ['id', 'email', 'company_name', 'last_name', 'first_name', 'name', 'group', 'phone',
              'address1', 'address2', 'city', 'state', 'zip', 'country', 'phone2', 'saddress1',
              'saddress2', 'scity', 'sstate', 'szip', 'scountry', 'sphone2']

"""
LandsDownUnder

number > id
phone no. > phone
fax no. > phone
e-mail > email
group description > group
street address > address1
address (line 2) > address2
address (line 3) > address3
zip code > zip
hw account > hw id
"""
lda.columns = ['id', 'name', 'phone', 'fax', 'email', 'group', 'address1', 'address2', 'address3', 'city',
               'state code', 'state', 'zip', 'country', 'web_site', 'hw id']

lda_des = lda.describe()
hw_des = hw.describe()

lda.info()
hw.info()


def normalize_name(a: str) -> list[str]:
    a = a.lower()  # convert to lower case
    a = re.sub(r'[^a-z0-9 ]', '', a)  # keep numbers and letters and space
    tokens = word_tokenize(a)  # separate words
    tokens = [word for word in tokens if word not in all_stopwords]  # remove stopwords
    return tokens


def normalize_address(a: str) -> list[str]:
    a = a.lower()  # convert to lower case
    a = re.sub(r'[^a-z0-9 ]', '', a)  # keep numbers and letters and space
    tokens = word_tokenize(a)  # separate words
    tokens = [word for word in tokens if word not in address_stopwords]  # remove stopwords
    return tokens


def normalize_number(a) -> str:
    a = re.sub(r'[^0-9]', '', a)  # keep only numbers
    return a


def normalize_email(a) -> list[str]:
    a = a.lower()  # convert to lower case
    a = re.sub(r'[^a-z0-9 ]', ' ', a)  # keep numbers and letters and space
    tokens = word_tokenize(a)  # separate words
    tokens = [word for word in tokens if word not in all_stopwords]  # remove stopwords
    return tokens


def remove_www(a):
    if type(a) == list and 'www' in a:
        a.remove('www')
    return a


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
lda['state code'] = lda['state code'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
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
hw['name'] = hw['name'].apply(lambda x: normalize_name(x) if pd.notnull(x) else x)
hw['group'] = hw['group'].apply(lambda x: normalize_name(x) if pd.notnull(x) else x)
hw['phone'] = hw['phone'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
hw['address1'] = hw['address1'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['address2'] = hw['address2'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['city'] = hw['city'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['state'] = hw['state'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['zip'] = hw['zip'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
hw['country'] = hw['country'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['phone2'] = hw['phone2'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
hw['saddress1'] = hw['saddress1'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['saddress2'] = hw['saddress2'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['scity'] = hw['scity'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['sstate'] = hw['sstate'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['szip'] = hw['szip'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)
hw['scountry'] = hw['scountry'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
hw['sphone2'] = hw['sphone2'].apply(lambda x: normalize_number(x) if pd.notnull(x) else x)


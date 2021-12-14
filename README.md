# Record linkage in two different datasets
This is a case study of a record linkage or fuzzy duplicates problem.

**p.s:** for privacy reasons all private data displayed here is fake and not part of the real datasets, 
although made to appear similar.

## Problem description
The online retailer [Lands down under](https://www.landsdownunder.com/) is changing their business management system
and as part of the migration process they want to clean and combine their inner and 3rd party customer datasets.

Part of the cleaning process is identifying duplicates within their inner dataset and matching the records from their 
inner dataset to the 3rd party dataset.

The finale result should be 2 Excel files containing the duplicates and linked records, each file should have "groups"
of records in it, where each group contains all records suspected to represent the same customer. 
The retailer will then manually inspect each group, mark it as correct or incorrect and merge the records.

To save the retailers time and resources, type 1 errors (false positives) have higher significance than type 2 errors
(false negatives) and should be avoided.

## The datasets

As mentioned above, there are two datasets relevant to the problem:

### LDU
This is the inner dataset of the retailer. It contains 5365 rows and 16 columns.

Those are the columns and their structure:

1. **ID -** the customer id. numeric type, starting at `100001`
2. **Name -** can be either a company name or private name, a string.
3. **Phone -** can be a string or int because the format is not uniform, and might or might not contain symbols.
4. **Fax -** same as phone but only exists for very few customers.
5. **Email -** string containing customers email
6. **Group -** type of customer. either wholesale, designer or retail.
7. **Address -** street address, string. example: `365 Oyster shell drive`
8. **Address 2-** continuation of street address
9. **Address 3-** continuation of street address
10. **City & State -** city in full name and state code. example: `Naples, FL`
11. **City -** same city in full name without state. Example: `Naples`
12. **State code -** same state code as before. Example: `FL`
13. **State -** state in full name. Example: `Florida`
14. **Zip -** Numeric or string, might contain letters. Example: `19925`
15. **Country -** county in full name, almost always. Example: `United States`
16. **Website -** link to companies page, and somtimes an email address

### HW
This is the 3rd party dataset of the retailer. It contains 3229 rows and 39 columns. 
I will only describe the ones that are not dropped during preprocessing.

Those are the columns and their structure:

1. **ID -** the customer id. numeric type, starting at `2`
2. **Name -** private name, a string.
3. **Name 2 -** private name or company name, a string.
4. **Company Name -** a string.
5. **Phone -** can be a string or int because the format is not uniform, and might or might not contain symbols.
6. **Email -** string containing customers email
7. **Group -** type of customer. either wholesale, designer or retail.
8. **Address -** street address, string. example: `365 Oyster shell drive`
9. **Address 2-** continuation of street address
10. **City -** same city in full name without state. Example: `Naples`
11. **State -** state in either full name or state code. Example: `Florida`
12. **Zip -** Numeric or string, might contain letters. Example: `19925`
13. **Country -** county in full name, almost always. Example: `United States`
14. **Phone 2 -** secondary phone.
15. **Address 3 -** shipping address in same format
16. **Address 4-** shipping address in same format
17. **City 2 -** shipping address in same format
18. **State 2 -** shipping address in same format
19. **Zip 2 -** shipping address in same format
20. **Country 2 -** shipping address in same format
21. **Phone 3 -** third phone related to the customer

## Preprocessing

First, we must match the datasets to each other as soon as possible, therefore, 
I dropped all columns from the HW dataset besides the ones mentioned above, that have an equivalent in the LDU dataset.

Next, I dropped all records with un-regular customer ids that are not true records according to the retailer and 
all records that have no data in them besides the id. This resulted in dropping 25 records from LDU and 0 from HW.

Now we must address the formatting. There are several problems with formatting in the datasets, basically we must make 
sure that all columns of the same type are in the same format and ready to be compared.

This is contains several stages:
1. remove symbols and punctuation from all columns. Example: `"abc@gmail.com"` -> `"abc gmail com"`
2. change all letters to lower case
3. tokenize all strings (divide to a list of sub strings) using `nltk.tokenize.words_tokenize()`. example: `"abc gmail com"` -> `["abc", "gmail", "com"]`
4. remove stop words from all columns using `nltk.corpus.stopwords`, stop words are words that contain no data and should be removed for better comparisons. stop words such as: street, and, www, the, a, an, ave....
5. sort names alphabetically to make sure identical names are in an identical order
6. change all states to full name using the [US](https://pypi.org/project/us/) library
7. change all countries to full name using the [country-converter](https://pypi.org/project/country-converter/) library
8. convert all empty strings and missing values to `numpy.nan`

Those actions were performed by defining a function that takes a value and returns it in the new format for each type 
of column and applying this function to the relevant columns here is an example for the address columns:
```python
def normalize_address(a: str):
    a = a.lower()  # convert to lower case
    a = re.sub(r'[^a-z0-9 ]', '', a)  # keep only numbers and letters and space
    tokens = word_tokenize(a)  # separate words
    tokens = [word for word in tokens if word not in stopwords]  # remove stopwords including: str, street, square...
    tokens.sort()  # sort alphabetically
    res = ' '.join(tokens)
    return res

# apply the function to the entire column except of missing values
ldu['address1'] = ldu['address1'].apply(lambda x: normalize_address(x) if pd.notnull(x) else x)
```

Next we deal with duplicate values in the same record. to simplify the process I decided that when the same phone 
number, name or address appear in the same record I will only keep one. 
This is not mandatory but makes the data cleaner and improves run times a bit.
```python
for i in hw.index:
    if hw.loc[i, 'address'] == hw.loc[i, 'address2']:
        hw.loc[i, 'address2'] = np.nan
```
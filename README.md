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
2. **first name -** private name, a string.
3. **last name -** private name, a string.
4. **Name 2 -** private name or company name, a string.
5. **Company Name -** a string.
6. **Phone -** can be a string or int because the format is not uniform, and might or might not contain symbols.
7. **Email -** string containing customers email
8. **Group -** type of customer. either wholesale, designer or retail.
9. **Address -** street address, string. example: `365 Oyster shell drive`
10. **Address 2-** continuation of street address
11. **City -** same city in full name without state. Example: `Naples`
12. **State -** state in either full name or state code. Example: `Florida`
13. **Zip -** Numeric or string, might contain letters. Example: `19925`
14. **Country -** county in full name, almost always. Example: `United States`
15. **Phone 2 -** secondary phone.
16. **Address 3 -** shipping address in same format
17. **Address 4-** shipping address in same format
18. **City 2 -** shipping address in same format
19. **State 2 -** shipping address in same format
20. **Zip 2 -** shipping address in same format
21. **Country 2 -** shipping address in same format
22. **Phone 3 -** third phone related to the customer

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
6. join list of tokens to single string with spaces 
7. change all states and countries to full name in lower case using the [US](https://pypi.org/project/us/) and [country-converter](https://pypi.org/project/country-converter/) libraries
8. convert all empty strings and missing values to `numpy.nan`

Those actions were performed by defining a function that takes a value and returns it in the new format for each type 
of column and applying this function to the relevant columns. Here is an example for the address columns:
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

One of the challenges is the different structure of the "same" field in the two datasets, for example in LDU there is 
one field "name" but in HW there are 4 equivalent fields, one of the ways to simplify it is to merge the "last name"
and "first name" fields in HW. We will do the same to the "address", "address2", "address3" fields in LDU.


Next we deal with duplicate values in the same record. to simplify the process I decided that when the same phone 
number, name or address appear in the same record I will only keep one. 
This is not mandatory but makes the data cleaner and improves run times a bit.
```python
for i in hw.index:
    if hw.loc[i, 'address'] == hw.loc[i, 'address2']:
        hw.loc[i, 'address2'] = np.nan
```

## Methodology
### Calculating similarity
As mentioned above, there are several challenges in this case that prevent usage of the usual methods such as the 
[RecordLinkage](https://recordlinkage.readthedocs.io/en/latest/about.html) or the [FuzzyWuzzy](https://pypi.org/project/fuzzywuzzy/)
packages:

* none matching fields
* very large amount of missing data
* over 17 million possible links

The first step will be creating a dataframe in which each row contains the similarity score for each field between the 
possible links.

Because of the none matching fields, at some cases, we cannot simply compare "name" to "name". Since we are only 
looking for matching values I decided to compare all fields to all fields of the same type, so for example,
we are comparing "city" from LDU both to "city" and "city2" in HW, than we will pick the highest similarity score 
between those comparisons and use it as the score for "city" similarity between the records.

Because there are very large amounts of missing values in both datasets:

```ldu.isna().sum()``` returns:

* id:             0
* name:           0
* phone:       1186
* fax:         5107
* email:        838
* group:       1082
* address:      387
* city:         378
* state:        428
* zip:          415
* country:      378
* web_site:    5193

I chose to simply ignore those values and assign them a similarity score of zero, in most cases this not a problem
because there is at least 1 value in the fields of the same type.

The runtime for comparing all the fields on all the possible links and generating the dataframe I specified above
is > 6 hours on my machine, so I decided to it into 10 parallel processes, each handling 1/10 of the possible links.

I used 2 different algorithms from the package [text distance](https://pypi.org/project/textdistance/) 
to calculate the similarity score.
For larger strings with several parts like email or address I used the `td.ratcliff_obershelp` and for strings with 
usually one word, I used `td.levenshtein.normalized_similarity`.

Both those algorithms accept 2 strings and return a value between 0 and 1 representing how similar they are,
0 = not similar at all and 1 = exactly the same.

Here is the top five rows of the generated dataframe, its shape is `(17788561, 14)`:

|     | index1 | index2 |    email | company_name |     name | group | phone | address | city | state | zip | country |   total |    score |
|----:|-------:|-------:|---------:|-------------:|---------:|------:|------:|--------:|-----:|------:|----:|--------:|--------:|---------:|
|   0 |      0 |      0 |        0 |            0 |        0 |     0 |     0 |       0 |    0 |     0 |   0 |       0 |       0 |        0 |
|   1 |      0 |      1 |    0.375 |            0 | 0.285714 |     1 |     0 |       0 |    0 |     0 |   0 |       0 | 1.66071 | 0.385714 |
|   2 |      0 |      2 | 0.325581 |            0 | 0.228571 |     1 |     0 |       0 |    0 |     0 |   0 |       0 | 1.55415 | 0.328571 |
|   3 |      0 |      3 | 0.322581 |            0 |     0.25 |     1 |     0 |       0 |    0 |     0 |   0 |       0 | 1.57258 |     0.35 |
|   4 |      0 |      4 | 0.410256 |            0 | 0.171429 |     1 |     0 |       0 |    0 |     0 |   0 |       0 | 1.58168 | 0.271429 |

index1 is the HW account index and index2 is the LDU account index. `total` is the sum of the row and `score` is a calculated
score I ended up not needing.

### Defining matches
Now that we have the similarity scores for all possible links, we must decide which ones will be considered as a match.
There are several possible approaches to this:
1. formulating a combined score and select its few top percents. I did this and found it to not work as well as the other methods. a combined score could be: `email * 0.3 + name * 0.9` and so on.
2. using the sum of all values in a row and deciding according to that. this approach has one major downside, the amount of missing data could easily overshadow the importance of good scores, for example if both email and name are an exact match than we probably have a match but if there are a lot of missing values we will still end up with a low total score.
3. defining specific scenarios which will be considered a match.

I ended up using the 3rd method and defined the scenarios as follows:
* `(name >= 0.8 or company_name >= 0.9) and (email >= 0.9 or phone >= 0.9)`
* `city >= 0.9 and state >= 0.9 and zip >= 1.0 and country >= 1.0 and address >= 1.0`
* `city >= 0.8 and state >= 0.8 and zip >= 0.8 and country >= 0.9 and phone >= 1.0`
* `name >= 1.0 or company_name >= 1.0 or email >= 1.0`

To get those values I plotted the distribution of each field, and depending on the distribution  chose the threshold.
for example:
![](https://github.com/Tommer-R/LandsDownUnder-record-linkage/blob/main/plots/state%20plot.png)
As you can see, there is a **large** number of very high scores, so we could use a threshold of 1.0.
That being said, for address matches we use 5 different fields, so we can afford to use a slightly lower threshold of 0.9
because that will allow for small changes like typos.

Most of the matches found exist in more than one of scenarios but each scenario also give us a few unique matches.
In total, we identified 2089 matches.

But what if one record in HW matches to 3 different records in LDU? this probably means that they all represent the same customer.
Therefore, the next stage is grouping the matches we found.

After grouping we end up with 1674 groups, of which, the longest group contains 8 different records.

### LDU duplicates
The goal was not only finding the matches between the two datasets but also finding duplicates inside the LDU dataset.

This process is simpler since there are no problems with the non-equivalent columns, therefore, I used the exact same 
method as specified above but with slightly different thresholds. 

As a result I found 274 duplicates which were divided into 184 groups.

The mush smaller number of results is expected since ideally, there shouldn't be any duplicates to begin with but there 
should be a large number of matches between the two datasets.

## Conclusion
This is a variation of a very common problem many businesses deal with. The method above successfully found the majority 
of matches and duplicates with a minimal number of false positives.

With relatively minor changes, this method could be applied to any 2 datasets.
If you do wish to implement it yourself, here are some tips you might find useful:
* make sure each column in your datasets contains as much unique information as possible and as little extras as possible
* remove unnecessary words such as: LLC, inc, street, ave, square, st, and so on...
* the column's format should be uniform, for example always numbers first and then words
* logically match different columns to each other assuming human error while filling in the data, for example: shipping address could be compared both with normal address and shipping address
* for especially large datasets you can use only a part of the possible links to save run time, for example check only those with matching countries.
* while deciding on the correct scenarios and thresholds constantly check the results and adjust to prevent recurring problems
* look for unreasonably large groups, this is an indication for large number of false positives

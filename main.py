import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from nltk.stem.snowball import SnowballStemmer


# from nltk.corpus import wordnet
import spacy
# import contextualSpellCheck
import time
from spacy.tokenizer import Tokenizer
import re


def readCSV(fileName):
    df = pd.read_csv(fileName, encoding='iso-8859-1')
    return df


att = readCSV('./data/attributes.csv')
product_desc = readCSV('./data/product_descriptions.csv')
sampleSub = readCSV('./data/sample_submission.csv')
trainData = readCSV('./data/train.csv')
testData = readCSV('./data/test.csv')
globalCount = 0
start = time.time()


class Week6:

    def __init__(self):
        pass

    def evaluation_RSME(self):
        # 1. Make a 80-20 split of the training set, using 80% for training and 20% for testing using the train_test_split function in sklearn")

        # 2. Evaluate the predictions on the test set in terms of Root Mean Squared Error (RMSE). Verify that your result is close to 0.48
        stemmer = SnowballStemmer('english')

        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")
        df_pro_desc = pd.read_csv(
            './data/product_descriptions.csv')

        def str_stemmer(s):
            return " ".join([stemmer.stem(word) for word in s.lower().split()])

        def str_common_word(str1, str2):
            return sum(int(str2.find(word) >= 0) for word in str1.split())

        df_all = df_train

        df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

        df_all['search_term'] = df_all['search_term'].map(
            lambda x: str_stemmer(x))
        df_all['product_title'] = df_all['product_title'].map(
            lambda x: str_stemmer(x))
        df_all['product_description'] = df_all['product_description'].map(
            lambda x: str_stemmer(x))

        df_all['len_of_query'] = df_all['search_term'].map(
            lambda x: len(x.split())).astype(np.int64)

        df_all['product_info'] = df_all['search_term']+"\t" + \
            df_all['product_title']+"\t"+df_all['product_description']

        df_all['word_in_title'] = df_all['product_info'].map(
            lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
        df_all['word_in_description'] = df_all['product_info'].map(
            lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))

        pd.DataFrame({"id": df_all['id'], "product_uid": df_all['product_uid'], "word_in_title": df_all['word_in_title'],
                     "word_in_description": df_all['word_in_description'], "len_of_query": df_all["len_of_query"]}).to_csv('filtered_default.csv', index=False)
        
        relevanceScore = df_all['relevance']
        df_all = df_all.drop(['search_term', 'product_title',
                             'product_description', 'product_info'], axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df_all, relevanceScore, test_size=0.2, train_size=0.8)

        X_train = X_train.drop(['id', 'relevance'], axis=1).values
        X_test = X_test.drop(['id', 'relevance'], axis=1).values

        rf = RandomForestRegressor(
            n_estimators=15, max_depth=6, random_state=0)
        clf = BaggingRegressor(rf, n_estimators=45,
                               max_samples=0.1, random_state=25)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(mean_squared_error(y_test, y_pred, squared=False))

    def evaluation_no_stremmer(self):
        # 3. Evaluate the matching without stemming for search terms, product titles, and product descriptions.
        stemmer = SnowballStemmer('english')

        df_train = pd.read_csv('./data/train.csv', encoding="ISO-8859-1")
        df_pro_desc = pd.read_csv('./data/product_descriptions.csv')

        def str_stemmer(s):
            return " ".join([stemmer.stem(word) for word in s.lower().split()])

        def str_common_word(str1, str2):
            return sum(int(str2.find(word) >= 0) for word in str1.split())

        df_all = df_train

        df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

        df_all['len_of_query'] = df_all['search_term'].map(
            lambda x: len(x.split())).astype(np.int64)

        df_all['product_info'] = df_all['search_term']+"\t" + \
            df_all['product_title']+"\t"+df_all['product_description']

        df_all['word_in_title'] = df_all['product_info'].map(
            lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
        df_all['word_in_description'] = df_all['product_info'].map(
            lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))

        relevanceScore = df_all['relevance']
        df_all = df_all.drop(['search_term', 'product_title',
                             'product_description', 'product_info', 'relevance'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            df_all, relevanceScore, test_size=0.2, train_size=0.8)

        rf = RandomForestRegressor(
            n_estimators=15, max_depth=6, random_state=0)
        clf = BaggingRegressor(rf, n_estimators=45,
                               max_samples=0.1, random_state=25)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # RMSE value increases from 0.48 to 0.51 (thus performing worse).
        print(mean_squared_error(y_test, y_pred, squared=False))

        # 2) Improving the match
        # Added word count with attribute values (based on bullet info)
        stemmer = SnowballStemmer('english')

        df_train = pd.read_csv('./data/train.csv', encoding="ISO-8859-1")
        df_pro_desc = pd.read_csv('./data/product_descriptions.csv')
        df_att = pd.read_csv('./data/attributes.csv')

        # only get rows with id that is in train df.
        all_id_df = pd.merge(df_train.drop_duplicates(
            subset=['product_uid']), df_att, how='left', on='product_uid')
        # all_id_df = all_id_df[all_id_df['name'].str.contains(
        #     "Bullet", na=False)]
        all_id_df = all_id_df.drop(
            ['search_term', 'product_title', 'name', 'relevance'], axis=1)
        test = all_id_df.groupby(['product_uid'])

        # df_all_att contains all att info for each product id
        df_all_att = pd.DataFrame()
        for product in test:
            row_to_append = pd.DataFrame([{'product_uid': product[0], 'bullet_info': " ".join(
                product[1].loc[:, 'value'].to_numpy(dtype=str))}])
            df_all_att = pd.concat([df_all_att, row_to_append])

        def str_stemmer(s):
            return " ".join([stemmer.stem(word) for word in str(s).lower().split()])

        def str_common_word(str1, str2):
            return sum(int(str2.find(word) >= 0) for word in str1.split())

        df_all = df_train

        df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
        df_all = pd.merge(df_all, df_all_att, how='left', on='product_uid')

        print(df_all.head(10))
        df_all['search_term'] = df_all['search_term'].map(
            lambda x: str_stemmer(x))
        df_all['product_title'] = df_all['product_title'].map(
            lambda x: str_stemmer(x))
        df_all['product_description'] = df_all['product_description'].map(
            lambda x: str_stemmer(x))
        df_all['bullet_info'] = df_all['bullet_info'].map(
            lambda x: str_stemmer(x))

        df_all['len_of_query'] = df_all['search_term'].map(
            lambda x: len(x.split())).astype(np.int64)

        df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] + \
            "\t"+df_all['product_description']+"\t"+df_all['bullet_info']

        df_all['word_in_title'] = df_all['product_info'].map(
            lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
        df_all['word_in_description'] = df_all['product_info'].map(
            lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
        df_all['word_in_bullets'] = df_all['product_info'].map(
            lambda x: str_common_word(x.split('\t')[0], x.split('\t')[3]))

        relevanceScore = df_all['relevance']
        df_all = df_all.drop(['search_term', 'product_title',
                             'product_description', 'product_info', 'bullet_info', 'relevance'], axis=1)
        print(df_all.head(20))
        X_train, X_test, y_train, y_test = train_test_split(
            df_all, relevanceScore, test_size=0.2, train_size=0.8)

        rf = RandomForestRegressor(
            n_estimators=15, max_depth=6, random_state=0)
        clf = BaggingRegressor(rf, n_estimators=45,
                               max_samples=0.1, random_state=25)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # RMSE value from 0.48 to 0.48 (Again no improvement)
        print(mean_squared_error(y_test, y_pred, squared=False))

    def generate_brand_matching_feature(self):
        """
        The follow features are implemented:
        1. Count number of brand name words matching with the query
        """
        df_filtered_data = pd.read_csv('./filtered_train.csv',encoding="ISO-8859-1")
        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")
        df_att = pd.read_csv('./data/attributes.csv')
        
        stemmer = SnowballStemmer('english')

        
        all_id_df = pd.merge(df_train.drop_duplicates(
            subset=['product_uid']), df_att, how='left', on='product_uid')

        test = all_id_df.groupby(['product_uid'])

        # df_brand_names contain
        df_brand_names = pd.DataFrame()
        for product in test:
            row_to_append = pd.DataFrame([{'product_uid': product[0], 'brand_name': " ".join(product[1].loc[product[1]['name'] == 'MFG Brand Name'].loc[:,'value'].to_numpy(dtype=str))}])
            df_brand_names = pd.concat([df_brand_names, row_to_append])
        
        merged = pd.merge(df_train, df_brand_names, how='left', on='product_uid')
        merged = merged.drop(
            ['product_uid', 'id','relevance'], axis=1)
              

        def found_common_words(query, brand_name, stemmer):
            if (brand_name == "Unbranded"): #unbranded product
                # id - 101069
                return 0,0, ''

            brandNameInQuery = 0 # brand word matching counter
            sizeBrandName = len(brand_name.split())

            queryWords = query.lower().split()

            brandNameQuery = [] # stores the found query words that match the brand name(s).

            # find brand word in title
            for index, queryWord in enumerate(queryWords):
                # Query word is in title at index [indexWord]
                for index2, titleWord in enumerate(brand_name.lower().split()):
                # check if first two letters are equal
                    if (queryWord[:2] == titleWord[:2]):
                        # count percentage of letters in the word.
                        count = 0
                        for char in queryWord:
                            if char in titleWord:
                                count += 1
                        # should atleast containt 75% of the letters of the query word.
                        if ((count/len(queryWord)) > 0.75):
                            # Check that found title word is not twice as long 
                            if ((len(queryWord)*2) > len(titleWord)):
                                # check dependency of word for the query
                                brandNameQuery.append(queryWords[index])
                                brandNameInQuery += 1
                                break
                                
            # Total run time is around 1 minute
            global globalCount
            globalCount += 1
            if ((globalCount % 1000) == 0):
                print("globalCount", globalCount)
                global start
                print("Elapsed ", time.time() - start)
            
            return brandNameInQuery,sizeBrandName, " ".join([word for word in brandNameQuery])
        
        result = merged.apply(lambda row:found_common_words(row['search_term'],row['brand_name'], stemmer),axis=1)
        y = result.to_frame()

        df_filtered_data['brand_in_query'] = y[0].map(lambda x: x[0])
        df_filtered_data['brand_name_size'] = y[0].map(lambda x: x[1])
        df_filtered_data['brand_name_query'] = y[0].map(lambda x: x[2])
        
        # Stores the features as filtered_brand_name.csv
        pd.DataFrame({"id": df_filtered_data['id'], "product_uid": df_filtered_data['product_uid'], "brand_in_query": df_filtered_data['brand_in_query'],"brand_name_size":df_filtered_data['brand_name_size'],"brand_name_query":df_filtered_data['brand_name_query']}).to_csv('filtered_brand_name.csv', index=False)

    def generate_title_desc_query_match(self):
        """
        The follow features are generated:
        1. Count length of query
        2. count words that is in the title of the product.
        --- count number of root in query
        --- count number of compounds in query
        3. count words that is in the description of the product.
        --- count number of root in query
        --- count number of compounds in query
        """
        
        nlp = spacy.load("en_core_web_lg")
        # Tokenize only on spaces.
        nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
        stemmer = SnowballStemmer('english')
        
        filtered_brand_name = pd.read_csv('./filtered_brand_name.csv',encoding="ISO-8859-1")
        df_brand = filtered_brand_name[(filtered_brand_name['brand_in_query'] == filtered_brand_name['brand_name_size']) & (filtered_brand_name['brand_name_size'] > 0)]
        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")#,nrows= 100)
        df_pro_desc = pd.read_csv('./data/product_descriptions.csv')

        def found_common_words(searchTerm, title, description, nlp, stemmer):
            # Count number of Syntactic dependency in title or description
            queryRootInTitle = 0
            queryRootInDesc = 0
            queryCompoundInTitle = 0
            queryCompoundInDesc = 0
            queryOtherInTitle = 0
            queryOtherInDesc = 0
            totalQueryRoot = 0
            totalQueryCompound = 0
            totalQueryOther = 0
            rootTermsTitle = []
            compoundTermsTitle = []
            otherTermsTitle = []
            rootTermsDesc = []
            compoundTermsDesc = []
            otherTermsDesc = []

            # Counts whether the root of the query is also the root in the title/description.
            # Term in query can be root, but does not have to be root in Title/description
            queryRootAlsoTitleRoot = 0
            queryRootAlsoDescRoot = 0

            # remove ? and $ from query
            query = searchTerm.replace("?","")
            query = query.replace("$","")
            query = query.strip()


            # Syntactic dependency, i.e. the relation between tokens for each term in query
            depQuery = []

            for token in nlp(query):
                tokenDep = token.dep_
                if (tokenDep == 'ROOT'):
                    totalQueryRoot += 1
                elif (tokenDep == 'compound'):
                    totalQueryCompound += 1
                else:
                    totalQueryOther += 1
                depQuery.append(tokenDep)

            depTitle = []

            for token in nlp(title):
                depTitle.append(token.dep_)

            # List which will hold all unfound query terms
            missingQueryWords = query.lower().split()

            # list that containes all query words stremmed
            queryStremmed = [stemmer.stem(word)
                             for word in query.lower().split()]

            # list that containes all title words stremmed
            titleStremmed = [stemmer.stem(word)
                             for word in title.lower().split()]

            # list used to store to be removed indices (these are found)
            tempIndices = []

            # find query terms in title
            for index, queryWord in enumerate(queryStremmed):
                # Query word is in title at index [indexWord]
                for index2, titleWord in enumerate(titleStremmed):
                    if queryWord == titleWord:
                        # check dependency of word
                        dep = depQuery[index]

                        # Check if query word occures multiple times in the title
                        indices = [i for i, x in enumerate(titleStremmed) if x == queryWord]
                        if (dep == 'ROOT'): # word is a root word
                            queryRootInTitle += 1
                            rootTermsTitle.append(missingQueryWords[index])
                            # found query word multiple times in the title
                            for pos in indices:
                                if (depTitle[pos] == dep):
                                    queryRootAlsoTitleRoot += 1
                                    break
                        elif (dep == 'compound'): # word is a compound word
                            queryCompoundInTitle += 1
                            compoundTermsTitle.append(missingQueryWords[index])
                        else: # word is something else
                            queryOtherInTitle += 1
                            otherTermsTitle.append(missingQueryWords[index])
                        
                        # Found index must be removed (later) from the query
                        tempIndices.append(index)
                        break

            tempIndices.reverse()
            for i in tempIndices: # removed query words that have been find thusfar.
                del depQuery[i]
                del missingQueryWords[i]
            tempIndices = []

            # We still have words which we have not found
            if (len(missingQueryWords) > 0):
                # Check if stremmed is in description
                depDesc = []

                for token in nlp(description):
                    depDesc.append(token.dep_)

                # list that contained all missing query words stremmed
                queryStremmed = [stemmer.stem(word)
                                 for word in missingQueryWords]

                # list that containes all title words stremmed
                descStremmed = [stemmer.stem(word)
                                for word in description.lower().split()]

                # find query terms in description
                for index, queryWord in enumerate(queryStremmed):
                    # Query word is in description
                    for index2, descWord in enumerate(descStremmed):
                        if queryWord == descWord:
                            # check dependency of word for the query
                            dep = depQuery[index]
                            if (dep == 'ROOT'):
                                queryRootInDesc += 1
                                rootTermsDesc.append(missingQueryWords[index])
                                if (depDesc[index2] == dep):
                                    queryRootAlsoDescRoot += 1
                            elif (dep == 'compound'):
                                compoundTermsDesc.append(
                                    missingQueryWords[index])
                                queryCompoundInDesc += 1
                            else:
                                otherTermsDesc.append(missingQueryWords[index])
                                queryOtherInDesc += 1

                            # remove word from the query
                            tempIndices.append(index)
                            break
                tempIndices.reverse()
                for i in tempIndices:
                    del depQuery[i]
                    del missingQueryWords[i]

                tempIndices = []

            # We still have words which we have not found
            if (len(missingQueryWords) > 0):
                # Check if misspelled is in title
                for index, queryWord in enumerate(missingQueryWords.copy()):
                    for index2, titleWord in enumerate(title.lower().split()):
                        # check if first two letters are equal
                        if (queryWord[:2] == titleWord[:2]):
                            # count percentage of letters in the word.
                            count = 0
                            for char in queryWord:
                                if char in titleWord:
                                    count += 1
                            # should atleast containt 75% of the letters of the query word.
                            if ((count/len(queryWord)) > 0.75):
                                # print("more than 75 percent")
                                # Check that found title word is not twice as long 
                                if ((len(queryWord)*2) > len(titleWord)):
                                    # check dependency of word for the query
                                    dep = depQuery[index]
                                    if (dep == 'ROOT'):
                                        queryRootInTitle += 1
                                        rootTermsTitle.append(
                                            missingQueryWords[index])
                                        if (depTitle[index2] == dep):
                                            queryRootAlsoTitleRoot += 1
                                    elif (dep == 'compound'):
                                        compoundTermsTitle.append(
                                            missingQueryWords[index])
                                        queryCompoundInTitle += 1
                                    else:
                                        otherTermsTitle.append(
                                            missingQueryWords[index])
                                        queryOtherInTitle += 1
                                    # remove word from the query
                                    tempIndices.append(index)
                                    break
                
                tempIndices.reverse()
                for i in tempIndices:
                    del depQuery[i]
                    del missingQueryWords[i]

                tempIndices = []
            # Run time is around 25 minutes for 75k enteries
            global globalCount
            globalCount += 1
            if ((globalCount % 1000) == 0):
                print("globalCount", globalCount)
                global start
                print("Elapsed ", time.time() - start)
            
            return queryRootInTitle, queryRootInDesc, queryCompoundInTitle, queryCompoundInDesc, queryOtherInTitle, queryOtherInDesc, totalQueryRoot, totalQueryCompound, totalQueryOther, queryRootAlsoTitleRoot, queryRootAlsoDescRoot

        df_all = df_train
        df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
        df_all_w_brand = pd.merge(df_brand, df_all, how='left', on=['id','product_uid'])
        df_all = df_all[~df_all.id.isin(df_all_w_brand.id)]

        def removeBrand(query, brand_name):
            words = brand_name.lower().split()
            return ' '.join([term for term in query.lower().split() if term not in words])
        
        df_all_w_brand['search_term'] = df_all_w_brand.apply(lambda row: removeBrand(row['search_term'], row['brand_name_query']),axis=1)
        print(df_all_w_brand)
        # Brand names are now removed from the query
        df_brand_names = pd.concat([df_all, df_all_w_brand], axis=0)
        
        df_brand_names = df_brand_names.sort_values(by=['id'])
        df_brand_names = df_brand_names.drop(['brand_in_query', 'brand_name_size','brand_name_query'], axis=1)
        print(df_brand_names)
        df_brand_names['product_info'] = df_brand_names['search_term']+"\t" + \
            df_brand_names['product_title']+"\t"+df_brand_names['product_description']\
             
        x = df_brand_names['product_info'].map(lambda x: found_common_words(
            x.split('\t')[0], x.split('\t')[1], x.split('\t')[2], nlp, stemmer))
        y = x.to_frame()
        # Add all the features.
        df_brand_names['query_root_in_title'] = y['product_info'].map(lambda x: x[0])
        df_brand_names['query_root_in_desc'] = y['product_info'].map(lambda x: x[1])
        df_brand_names['query_compound_in_title'] = y['product_info'].map(
            lambda x: x[2])
        df_brand_names['query_compound_in_desc'] = y['product_info'].map(
            lambda x: x[3])
        df_brand_names['query_other_in_title'] = y['product_info'].map(lambda x: x[4])
        df_brand_names['query_other_in_desc'] = y['product_info'].map(lambda x: x[5])
        df_brand_names['total_query_root'] = y['product_info'].map(lambda x: x[6])
        df_brand_names['total_query_compound'] = y['product_info'].map(lambda x: x[7])
        df_brand_names['total_query_other'] = y['product_info'].map(lambda x: x[8])
        df_brand_names['query_root_also_root_in_title'] = y['product_info'].map(
            lambda x: x[9])
        df_brand_names['query_compound_also_compound_in_title'] = y['product_info'].map(
            lambda x: x[10])

        # store the featuers in 'filtered_train.csv'
        pd.DataFrame({"id": df_brand_names['id'], "product_uid": df_brand_names['product_uid'], "query_root_in_title": df_brand_names['query_root_in_title'],
                     "query_ro  ot_in_desc": df_brand_names['query_root_in_desc'], "query_compound_in_title": df_brand_names["query_compound_in_title"], "query_compound_in_desc": df_brand_names["query_compound_in_desc"], "query_other_in_title": df_brand_names["query_other_in_title"], "query_other_in_desc": df_brand_names["query_other_in_desc"], "total_query_root": df_brand_names["total_query_compound"],"total_query_compound": df_brand_names["total_query_root"],"total_query_other": df_brand_names["total_query_other"],"query_root_also_root_in_title": df_brand_names["query_root_also_root_in_title"],"query_compound_also_compound_in_title": df_brand_names["query_compound_also_compound_in_title"],}).to_csv('filtered_train.csv', index=False)
        
    def generate_attribute_query_match(self):
        """
        The follow features are generated:
        1. count query words that match with the attribute values of the product.
        --- count number of root in query
        --- count number of compounds in query
        """
        df_filtered_data = pd.read_csv('./filtered_train.csv')
        df_filtered_data = df_filtered_data[['product_uid','id']]
        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")
        df_att = pd.read_csv('./data/attributes.csv')

        nlp = spacy.load("en_core_web_lg")
        # Tokenize only on spaces.
        nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
        # contextualSpellCheck.add_to_pipe(nlp)
        stemmer = SnowballStemmer('english')

        all_id_df = pd.merge(df_filtered_data.drop_duplicates(
            subset=['product_uid']), df_att, how='left', on='product_uid')
        test = all_id_df.groupby(['product_uid'])

        # df_all_att contains all attribute values for each product id
        df_all_att = pd.DataFrame()
        for product in test:
            row_to_append = pd.DataFrame([{'product_uid': product[0], 'att_info': ' '.join(
                product[1].loc[:, 'value'].to_numpy(dtype=str))}])
            df_all_att = pd.concat([df_all_att, row_to_append])
        
        merged = pd.merge(df_train, df_all_att, how='left', on='product_uid')       

        def found_common_words(searchTerm, attr_info, nlp, stemmer):
            # Count number of Syntactic dependency in title or description
            queryRootInAttr= 0
            queryCompoundInAttr = 0
            queryOtherInAttr = 0
            totalQueryRoot = 0
            totalQueryCompound = 0
            totalQueryOther = 0
            rootTermsTitle = []
            compoundTermsTitle = []
            otherTermsTitle = []

            query = searchTerm.replace("?","")
            query = query.replace("$","")
            query = query.strip()

            # Syntactic dependency, i.e. the relation between tokens for each term in query
            depQuery = []
            

            for token in nlp(query):
                tokenDep = token.dep_
                if (tokenDep == 'ROOT'):
                    totalQueryRoot += 1
                elif (tokenDep == 'compound'):
                    totalQueryCompound += 1
                else:
                    totalQueryOther += 1
                depQuery.append(tokenDep)

            depTitle = []

            for token in nlp(attr_info):
                depTitle.append(token.dep_)

            # List which will hold all unfound query terms
            missingQueryWords = query.lower().split()

            # list that containes all query words stremmed
            queryStremmed = [stemmer.stem(word)
                             for word in query.lower().split()]

            # list that containes all title words stremmed
            attrInfoStremmed = [stemmer.stem(word)
                             for word in attr_info.lower().split()]

            tempIndices = []
            # find query terms in title
            for index, queryWord in enumerate(queryStremmed):
                # Query word is in title at index [indexWord]
                for index2, attrWord in enumerate(attrInfoStremmed):

                    if queryWord == attrWord:
                        # check dependency of word
                        dep = depQuery[index]

                        if (dep == 'ROOT'):
                            queryRootInAttr += 1
                            rootTermsTitle.append(missingQueryWords[index])

                        elif (dep == 'compound'):
                            queryCompoundInAttr += 1
                            compoundTermsTitle.append(missingQueryWords[index])

                        else:
                            queryOtherInAttr += 1
                            otherTermsTitle.append(missingQueryWords[index])

                        # remove word from the query
                        tempIndices.append(index)
                        break

            tempIndices.reverse()
            for i in tempIndices:
                del depQuery[i]
                del missingQueryWords[i]
            tempIndices = []

            # We still have words which we have not found
            if (len(missingQueryWords) > 0):
                # Check if misspelled is in attr
                for index, queryWord in enumerate(missingQueryWords.copy()):
                    for index2, attrWord in enumerate(attr_info.lower().split()):
                        # check if first two letters are equal
                        if (queryWord[:2] == attrWord[:2]):
                            # count percentage of letters in the word.
                            count = 0
                            for char in queryWord:
                                if char in attrWord:
                                    count += 1
                            # should atleast containt 75% of the letters of the query word.
                            if ((count/len(queryWord)) > 0.75):
                                # Check that found title word is not twice as long 
                                if ((len(queryWord)*2) > len(attrWord)):
                                    # check dependency of word for the query
                                    dep = depQuery[index]
                                    if (dep == 'ROOT'):
                                        queryRootInAttr += 1
                                        rootTermsTitle.append(
                                            missingQueryWords[index])
                                    elif (dep == 'compound'):
                                        compoundTermsTitle.append(
                                            missingQueryWords[index])
                                        queryCompoundInAttr += 1
                                    else:
                                        otherTermsTitle.append(
                                            missingQueryWords[index])
                                        queryOtherInAttr += 1
                                    # remove word from the query
                                    tempIndices.append(index)
                                    break
                
                tempIndices.reverse()
                for i in tempIndices:
                    del depQuery[i]
                    del missingQueryWords[i]

                tempIndices = []
            
            # Generating feature set takes < 20 minutes
            global globalCount
            globalCount += 1
            if ((globalCount % 1000) == 0):
                print("globalCount", globalCount)
                global start
                print("Elapsed ", time.time() - start)

            return queryRootInAttr, queryCompoundInAttr, queryOtherInAttr,totalQueryRoot, totalQueryCompound, totalQueryOther
        
        result = merged.apply(lambda row:found_common_words(row['search_term'],row['att_info'], nlp, stemmer),axis=1)
        y = result.to_frame()

        df_filtered_data['query_root_in_attr'] = y[0].map(lambda x: x[0])
        df_filtered_data['query_compound_in_attr'] = y[0].map(lambda x: x[1])
        df_filtered_data['query_other_in_attr'] = y[0].map(lambda x: x[2])
        
        # store the feature in 'filtered_attr.csv'
        pd.DataFrame({"id": df_filtered_data['id'], "product_uid": df_filtered_data['product_uid'], "query_root_in_attr": df_filtered_data['query_root_in_attr'],"query_compound_in_attr":df_filtered_data['query_compound_in_attr'],"query_other_in_attr":df_filtered_data['query_other_in_attr']}).to_csv('filtered_attr.csv', index=False)

    def generateMeasurementfeatures(self):
        """
        The follow features are generated:
        1. count query words that match with the attribute values of the product.
        --- count number of root in query
        --- count number of compounds in query
        """
        df_filtered_data = pd.read_csv('./filtered_train.csv')
        df_filtered_data = df_filtered_data[['product_uid','id']]
        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")
        df_att = pd.read_csv('./data/attributes.csv') 

        # 1) find measurements in query

        # 2) if a measurement is found, check if it exists in attributes.
        

    def execute_model(self):
        df_filtered_data = pd.read_csv('./filtered_train.csv',encoding="ISO-8859-1")
        df_data = pd.read_csv('./data/train.csv',encoding="ISO-8859-1")
        df_filtered_default = pd.read_csv('./filtered_default.csv',encoding="ISO-8859-1")
        
        df_filtered_data['word_in_title'] = df_filtered_default['word_in_title']
        df_filtered_data['word_in_description'] = df_filtered_default['word_in_description']
        df_filtered_data['len_of_query'] = df_filtered_default['len_of_query']
        relevanceScore = df_data['relevance'].values

        df_filtered_data = df_filtered_data.drop(
            ['id', 'product_uid'], axis=1).values

        X_train, X_test, y_train, y_test = train_test_split(
            df_filtered_data, relevanceScore, test_size=0.2, train_size=0.8)
        
        rf = RandomForestRegressor(
            n_estimators=15, max_depth=6, random_state=0)
        clf = BaggingRegressor(rf, n_estimators=45,
                               max_samples=0.1, random_state=25)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
    
        print(mean_squared_error(y_test, y_pred, squared=False))

    def execute_model2(self):
        df_filtered_data = pd.read_csv('./filtered_default.csv',encoding="ISO-8859-1")
        df_data = pd.read_csv('./data/train.csv',encoding="ISO-8859-1")
        
        relevanceScore = df_data['relevance'].values
        df_filtered_data = df_filtered_data.drop(
            ['id', 'product_uid'], axis=1).values
        

        X_train, X_test, y_train, y_test = train_test_split(
            df_filtered_data, relevanceScore, test_size=0.2, train_size=0.8)
        
        rf = RandomForestRegressor(
            n_estimators=15, max_depth=6, random_state=0)
        clf = BaggingRegressor(rf, n_estimators=45,
                               max_samples=0.1, random_state=25)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        print(mean_squared_error(y_test, y_pred, squared=False))

    def execute_model3(self):
        df_filtered_data = pd.read_csv('./filtered_train.csv',encoding="ISO-8859-1")
        df_filtered_attr = pd.read_csv('./filtered_attr.csv',encoding="ISO-8859-1")
        df_data = pd.read_csv('./data/train.csv',encoding="ISO-8859-1")
        df_filtered_default = pd.read_csv('./filtered_default.csv',encoding="ISO-8859-1")

        # df_filtered_data = pd.merge(df_filtered_data, df_filtered_attr, how='left', on='id')
        # df_filtered_data = pd.merge(df_filtered_data, df_filtered_default, how='left', on='id')
        # # print(df_filtered_data.head(20))
        # # print(df_filtered_data.columns)
        # df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
        
        df_filtered_data['word_in_title'] = df_filtered_default['word_in_title']
        df_filtered_data['word_in_description'] = df_filtered_default['word_in_description']
        df_filtered_data['len_of_query'] = df_filtered_default['len_of_query']
        df_filtered_data['query_root_in_attr'] = df_filtered_attr['query_root_in_attr']
        df_filtered_data['query_compound_in_attr'] = df_filtered_attr['query_compound_in_attr']
        df_filtered_data['query_other_in_attr'] = df_filtered_attr['query_other_in_attr']
        relevanceScore = df_data['relevance'].values

        df_filtered_data = df_filtered_data.drop(
            ['id', 'product_uid'], axis=1).values

        X_train, X_test, y_train, y_test = train_test_split(
            df_filtered_data, relevanceScore, test_size=0.2, train_size=0.8)
        
        rf = RandomForestRegressor(
            n_estimators=15, max_depth=6, random_state=0)
        clf = BaggingRegressor(rf, n_estimators=45,
                               max_samples=0.1, random_state=25)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
    
        print(mean_squared_error(y_test, y_pred, squared=False))



class Week7:

    def __init__(self):
        pass

    def regression_models(self):
        X_train, y_train, X_test, y_test = self.getModifiedSplits()     # <- self.getSplits()

        # Naked models (all without bagging)
        # self.try_model(X_train, y_train, X_test, y_test, RandomForestRegressor())              # 0.559, t=21.99
        # self.try_model(X_train, y_train, X_test, y_test, KNeighborsRegressor())                # 0.570, t=0.11
        # self.try_model(X_train, y_train, X_test, y_test, HistGradientBoostingRegressor())      # 0.481, t=0.29
        # self.try_model(X_train, y_train, X_test, y_test, SVR())                                # 0.532, t=179
        
        self.try_model(X_train, y_train, X_test, y_test, HistGradientBoostingRegressor(
            **{'learning_rate': 0.12, 'max_depth': 15, 'max_iter': 400, 'min_samples_leaf': 17}
        )) #0.478, t=0.54
        

    def hyperparameter_optimization(self):
        """
        This function is used to GridSearch the optimal hyperparameters for HistGradientBoostingRegressor
        """
        X_train, y_train, _, _ = self.getModifiedSplits()       # <- self.getSplits()

        param_grid = {
            'learning_rate': [0.12],
            'max_depth': list(range(9,18,2)),
            'max_iter': list(range(400,750,50)),
            'min_samples_leaf': list(range(11,21,2)),
        }


        grid_search = GridSearchCV(HistGradientBoostingRegressor(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)


        print(grid_search.best_params_)
        print(grid_search.best_score_)

        with open("results.txt","w+") as f:
            f.write(str(grid_search.best_params_))

    def try_model(self, X_train, y_train, X_test, y_test, model, n_rep = 10):
        """
        Test the model with some training data to see what the error is
        """
        sum_error = 0
        sum_time = 0
        for i in range(n_rep):
            time1 = time.perf_counter()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            time2 = time.perf_counter()
            error = mean_squared_error(y_test, y_pred, squared=False)
            
            sum_error += error
            sum_time += time2-time1

        # Importance features

        importance = permutation_importance(model, np.concatenate((X_train,X_test),axis=0), np.concatenate((y_train,y_test),axis=0), n_repeats=25, random_state=0)
        print("feature importances",importance)

        df, _ = self.getModifiedDataset()
        df = df.drop(['id', 'product_uid'], axis=1)

        plt.bar(df.columns,importance['importances_mean'],yerr=importance['importances_std'])
        plt.xticks(rotation='vertical')
        plt.show()

        print("Performance of model:",type(model),f"n_rep={n_rep}")
        print("Average error:",sum_error/n_rep, "Average time", sum_time/n_rep)

        return sum_error/n_rep, sum_time/n_rep

    def getModifiedDataset(self):
        df_filtered_data = pd.read_csv('./filtered_train.csv',encoding="ISO-8859-1")
        df_filtered_attr = pd.read_csv('./filtered_attr.csv',encoding="ISO-8859-1")
        df_data = pd.read_csv('./data/train.csv',encoding="ISO-8859-1")
        df_brand_names = pd.read_csv('./filtered_brand_name.csv',encoding="ISO-8859-1")
        df_filtered_default = pd.read_csv('./filtered_default.csv',encoding="ISO-8859-1")
        
        df_filtered_data['word_in_title'] = df_filtered_default['word_in_title']
        df_filtered_data['word_in_description'] = df_filtered_default['word_in_description']
        df_filtered_data['len_of_query'] = df_filtered_default['len_of_query']
        df_filtered_data['query_root_in_attr'] = df_filtered_attr['query_root_in_attr']
        df_filtered_data['query_compound_in_attr'] = df_filtered_attr['query_compound_in_attr']
        df_filtered_data['query_other_in_attr'] = df_filtered_attr['query_other_in_attr']
        df_filtered_data['brand_in_query'] = df_brand_names['brand_in_query']
        df_filtered_data['brand_name_size'] = df_brand_names['brand_name_size']

        relevanceScore = pd.merge(df_filtered_data,df_data, how='left', on='id')
        relevanceScore = relevanceScore['relevance'].values

        # df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
        # df_filtered_data = df_filtered_data.drop(['query_root_in_desc', 
        #                                         'query_compound_in_desc',
        #                                         'query_other_in_desc',
        #                                         'total_query_root',
        #                                         'total_query_compound',
        #                                         'total_query_other',
        #                                         'query_root_also_root_in_title',
        #                                         'query_compound_also_compound_in_title',
        #                                         'query_root_in_attr',
        #                                         'query_compound_in_attr',
        #                                         'query_other_in_attr',
        #                                         'brand_in_query'], axis=1) # For time purposes, dropping irrelevant columns barely differs error, but differs significantly in time
        
        return df_filtered_data, relevanceScore

    def getModifiedSplits(self):
        df_filtered_data, relevanceScore = self.getModifiedDataset()
        df_filtered_data = df_filtered_data.drop(
            ['id', 'product_uid'], axis=1).values
        
        X_train, X_test, y_train, y_test = train_test_split(
            df_filtered_data, relevanceScore, test_size=0.2, train_size=0.8)
        
        return X_train, y_train, X_test, y_test
    
    def getSplits(self):
        """
        Get the datasets (Same as week 6a, but now in week 7 :p)
        """
        # Timer
        time1 = time.perf_counter()
        
        stemmer = SnowballStemmer('english')

        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")
        df_pro_desc = pd.read_csv(
            './data/product_descriptions.csv')

        def str_stemmer(s):
            return " ".join([stemmer.stem(word) for word in s.lower().split()])

        def str_common_word(str1, str2):
            return sum(int(str2.find(word) >= 0) for word in str1.split())

        df_all = df_train

        df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

        df_all['search_term'] = df_all['search_term'].map(
            lambda x: str_stemmer(x))
        df_all['product_title'] = df_all['product_title'].map(
            lambda x: str_stemmer(x))
        df_all['product_description'] = df_all['product_description'].map(
            lambda x: str_stemmer(x))

        df_all['len_of_query'] = df_all['search_term'].map(
            lambda x: len(x.split())).astype(np.int64)

        df_all['product_info'] = df_all['search_term']+"\t" + \
            df_all['product_title']+"\t"+df_all['product_description']

        df_all['word_in_title'] = df_all['product_info'].map(
            lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
        df_all['word_in_description'] = df_all['product_info'].map(
            lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))

        pd.DataFrame({"id": df_all['id'], "product_uid": df_all['product_uid'], "word_in_title": df_all['word_in_title'],
                     "word_in_description": df_all['word_in_description'], "len_of_query": df_all["len_of_query"]}).to_csv('filtered_default.csv', index=False)
        
        relevanceScore = df_all['relevance']
        df_all = df_all.drop(['search_term', 'product_title',
                             'product_description', 'product_info'], axis=1)
        # print(df_all.head(20))
        X_train, X_test, y_train, y_test = train_test_split(
            df_all, relevanceScore, test_size=0.2, train_size=0.8)

        X_train = X_train.drop(['id', 'relevance'], axis=1).values
        X_test = X_test.drop(['id', 'relevance'], axis=1).values

        return X_train, y_train, X_test, y_test
    
    
# Week5().dataExploration()

# print("Generating brand name feature")
# Week6().generate_brand_matching_feature()
# print("Generating title/desc query match feature")
# Week6().generate_title_desc_query_match()
# print("Generating attribute query match feature")
Week6().generate_attribute_query_match()

# Week6().execute_model3()
# Week6().execute_model2()

# Week6().evaluation_final()  

# Week6().evaluation_RSME()



# Week7().hyperparameter_optimization()
# Week6().generate_title_desc_query_match()
 
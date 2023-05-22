
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from nltk.stem.snowball import SnowballStemmer

import re
from spacy_wordnet.wordnet_annotator import WordnetAnnotator 
from nltk.corpus import wordnet
import spacy
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


class Features:

    def __init__(self):
        pass

    def generate_brand_matching_feature(self):
        # id 3453 contains only part of the title.
        # Extra column for whether a brand name exists.
        # Extra column that will tell if the positions of the brand name terms are correct.
        """
        The follow features are implemented:
        1. Count number of brand name words matching with the query
        """
        # df_filtered_data = pd.read_csv('./filtered_data/filtered_train.csv',encoding="ISO-8859-1")
        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")
        df_att = pd.read_csv('./data/attributes.csv', encoding="ISO-8859-1")
        
        # product 100109 does not have attribute date
        df_att = df_att.loc[df_att['name'] == 'MFG Brand Name']
        all_id_df = pd.merge(df_train, df_att, how='left', on='product_uid')

        all_without_brands = all_id_df.loc[all_id_df['name'] != 'MFG Brand Name'] # not all products have a brand name defined

        all_with_brands = all_id_df.loc[all_id_df['name'] == 'MFG Brand Name']

        all_without_brands = all_without_brands.drop(['name'], axis=1)
        all_without_brands = all_without_brands.fillna("")

        all_with_brands = all_with_brands.drop(['name'], axis=1)

        def found_common_words(query, brand_name):
            # print("query", query , "brand_name", brand_name)
            if (brand_name == "Unbranded" or isinstance(brand_name, float)): #unbranded product
                # id - 101069 = unbranded, 100329 = nan
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
        
        result = all_with_brands.apply(lambda row:found_common_words(row['search_term'],row['value']),axis=1)
        y = result.to_frame()
        
        all_without_brands["brand_name_query"] = ""
        all_with_brands['brand_length_in_query'] = y[0].map(lambda x: x[0])
        all_with_brands['brand_name_size'] = y[0].map(lambda x: x[1])
        all_with_brands['brand_name_query'] = y[0].map(lambda x: x[2])
        df_all = pd.merge(all_with_brands, all_without_brands, how='outer', on=['id','product_uid','search_term','relevance','value','product_title',"brand_name_query"])
        df_all = df_all.fillna(0)

        # Stores the features as filtered_brand_name.csv
        pd.DataFrame({"id": df_all['id'], "product_uid": df_all['product_uid'], "brand_length_in_query": df_all['brand_length_in_query'],"brand_name_size":df_all['brand_name_size'],"brand_name_query":df_all['brand_name_query'],"brand_name_in_attr":df_all['value']}).to_csv('filtered_brand_name.csv', index=False)


    def generate_title_query_match(self):
        """
        The follow features are generated:
        1. Count length of query
        2. count words that is in the title of the product.
        --- count number of root in query
        --- count number of compounds in query
        """
        
        nlp = spacy.load("en_core_web_lg")
        # Tokenize only on spaces.
        nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
        stemmer = SnowballStemmer('english')
        
        filtered_brand_name = pd.read_csv('./filtered_data/filtered_brand_name.csv',encoding="ISO-8859-1").fillna("")
        df_query_with_brands = filtered_brand_name[(filtered_brand_name['brand_length_in_query'] == filtered_brand_name['brand_name_size']) & (filtered_brand_name['brand_name_size'] > 0 )]
        df_query_with_brands = df_query_with_brands.loc[df_query_with_brands['brand_name_query'] != ""]

        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")
        df_pro_desc = pd.read_csv('./data/product_descriptions.csv', encoding="ISO-8859-1")

        def found_common_words(searchTerm, title, description, nlp, stemmer):
            # Count number of Syntactic dependency in title or description
            queryRootInTitle = 0
            queryCompoundInTitle = 0
            queryOtherInTitle = 0
            totalQueryRoot = 0
            totalQueryCompound = 0
            totalQueryOther = 0
            rootTermsTitle = []
            compoundTermsTitle = []
            otherTermsTitle = []


            # Counts whether the root of the query is also the root in the title/description.
            # Term in query can be root, but does not have to be root in Title/description
            queryRootAlsoTitleRoot = 0

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
            
            return queryRootInTitle, queryCompoundInTitle, queryOtherInTitle, totalQueryRoot, totalQueryCompound, totalQueryOther, queryRootAlsoTitleRoot, missingQueryWords

        df_all = df_train
        df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
        df_all_w_brand = pd.merge(df_query_with_brands, df_all, how='left', on=['id','product_uid'])

        df_all = df_all[~df_all.id.isin(df_all_w_brand.id)]

        def removeBrand(query, brand_name):
            words = brand_name.lower().split()
            if len(words) > 1:
                indices = []
                q = query.split()
                print(query)
                print(brand_name)
                for word_part in words:
                    indices.append(words.index(word_part))
                cpy = indices.copy()
                indices.sort()
                if (len(indices)-1 != (indices[-1] - indices[0])) or indices != cpy:
                    print("not correct position brand")
                    print((len(indices) != (indices[-1] - indices[0])))
                    print(indices != cpy)

                    print(indices)
                    print(cpy)
                    # exit()
                    print()
                    return query
                     
            return ' '.join([term for term in query.lower().split() if term not in words])
        
        df_all_w_brand['search_term'] = df_all_w_brand.apply(lambda row: removeBrand(row['search_term'], row['brand_name_query']),axis=1)
        # Brand names are now removed from the query
        df_brand_names = pd.concat([df_all, df_all_w_brand], axis=0)
        print(df_all_w_brand['id'][10])
        
        df_brand_names = df_brand_names.sort_values(by=['id'])
        df_brand_names = df_brand_names.drop(['brand_length_in_query','brand_name_in_attr', 'brand_name_size','brand_name_query'], axis=1)

        df_brand_names['product_info'] = df_brand_names['search_term']+"\t" + \
            df_brand_names['product_title']+"\t"+df_brand_names['product_description']\
             
        x = df_brand_names['product_info'].map(lambda x: found_common_words(
            x.split('\t')[0], x.split('\t')[1], x.split('\t')[2], nlp, stemmer))
        y = x.to_frame()

        # Add all the features.
        df_brand_names['query_root_in_title'] = y['product_info'].map(lambda x: x[0])
        df_brand_names['query_compound_in_title'] = y['product_info'].map(
            lambda x: x[1])
        df_brand_names['query_other_in_title'] = y['product_info'].map(lambda x: x[2])
        df_brand_names['total_query_root'] = y['product_info'].map(lambda x: x[3])
        df_brand_names['total_query_compound'] = y['product_info'].map(lambda x: x[4])
        df_brand_names['total_query_other'] = y['product_info'].map(lambda x: x[5])
        df_brand_names['query_root_also_root_in_title'] = y['product_info'].map(
            lambda x: x[6])
        df_brand_names['missing_query_terms'] = y['product_info'].map(
            lambda x: x[7])

        # store the featuers in 'filtered_train.csv'
        pd.DataFrame({"id": df_brand_names['id'], "product_uid": df_brand_names['product_uid'],"missing_query_terms": df_brand_names['missing_query_terms'], "query_root_in_title": df_brand_names['query_root_in_title'],
        "query_compound_in_title": df_brand_names["query_compound_in_title"], "query_other_in_title": df_brand_names["query_other_in_title"], "total_query_root": df_brand_names["total_query_root"],
        "total_query_compound": df_brand_names["total_query_compound"],"total_query_other": df_brand_names["total_query_other"],"query_root_also_root_in_title": df_brand_names["query_root_also_root_in_title"]}).to_csv('filtered_train_title.csv', index=False)
      

    def generate_desc_query_match(self):
        """
        The follow features are generated:
        1. Count length of query
        2. count words that is in the description of the product.
        --- count number of root in query
        --- count number of compounds in query
        """
        
        nlp = spacy.load("en_core_web_lg")
        # Tokenize only on spaces.
        nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
        stemmer = SnowballStemmer('english')
        
        filtered_brand_name = pd.read_csv('./filtered_data/filtered_brand_name.csv',encoding="ISO-8859-1").fillna("")
        df_query_with_brands = filtered_brand_name[(filtered_brand_name['brand_length_in_query'] == filtered_brand_name['brand_name_size']) & (filtered_brand_name['brand_name_size'] > 0 )]
        df_query_with_brands = df_query_with_brands.loc[df_query_with_brands['brand_name_query'] != ""]

        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")
        df_pro_desc = pd.read_csv('./data/product_descriptions.csv', encoding="ISO-8859-1")

        def found_common_words(searchTerm, title, description, nlp, stemmer):
            # Count number of Syntactic dependency in title or description
            queryRootInDesc = 0
            queryCompoundInDesc = 0
            queryOtherInDesc = 0
            totalQueryRoot = 0
            totalQueryCompound = 0
            totalQueryOther = 0
            rootTermsDesc = []
            compoundTermsDesc = []
            otherTermsDesc = []

            # Counts whether the root of the query is also the root in the title/description.
            # Term in query can be root, but does not have to be root in Title/description
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
                
            depDesc = []

            for token in nlp(description):
                depDesc.append(token.dep_)

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

            if (len(missingQueryWords) > 0):
                # Check if stremmed is in description
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
                    for index2, descWord in enumerate(descStremmed):
                        # check if first two letters are equal
                        if (queryWord[:2] == descWord[:2]):
                            # count percentage of letters in the word.
                            count = 0
                            for char in queryWord:
                                if char in descWord:
                                    count += 1
                            # should atleast containt 75% of the letters of the query word.
                            if ((count/len(queryWord)) > 0.75):
                                # print("more than 75 percent")
                                # Check that found title word is not twice as long 
                                if ((len(queryWord)*2) > len(descWord)):
                                    # check dependency of word for the query
                                    dep = depQuery[index]
                                    if (dep == 'ROOT'):
                                        queryRootInDesc += 1
                                        rootTermsDesc.append(
                                            missingQueryWords[index])
                                        if (depDesc[index2] == dep):
                                            queryRootAlsoDescRoot += 1
                                    elif (dep == 'compound'):
                                        compoundTermsDesc.append(
                                            missingQueryWords[index])
                                        queryCompoundInDesc += 1
                                    else:
                                        otherTermsDesc.append(
                                            missingQueryWords[index])
                                        queryOtherInDesc += 1
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
            
            return queryRootInDesc, queryCompoundInDesc, queryOtherInDesc, totalQueryRoot, totalQueryCompound, totalQueryOther, queryRootAlsoDescRoot, missingQueryWords

        df_all = df_train
        df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
        df_all_w_brand = pd.merge(df_query_with_brands, df_all, how='left', on=['id','product_uid'])

        df_all = df_all[~df_all.id.isin(df_all_w_brand.id)]

        def removeBrand(query, brand_name):
            words = brand_name.lower().split()
            if len(words) > 1:
                indices = []
                q = query.split()
                print(query)
                print(brand_name)
                for word_part in words:
                    indices.append(words.index(word_part))
                cpy = indices.copy()
                indices.sort()
                if (len(indices)-1 != (indices[-1] - indices[0])) or indices != cpy:
                    print("not correct position brand")
                    print((len(indices) != (indices[-1] - indices[0])))
                    print(indices != cpy)

                    print(indices)
                    print(cpy)
                    # exit()
                    print()
                    return query
                     
            return ' '.join([term for term in query.lower().split() if term not in words])
        
        df_all_w_brand['search_term'] = df_all_w_brand.apply(lambda row: removeBrand(row['search_term'], row['brand_name_query']),axis=1)
        # Brand names are now removed from the query
        df_brand_names = pd.concat([df_all, df_all_w_brand], axis=0)
        print(df_all_w_brand['id'][10])
        
        df_brand_names = df_brand_names.sort_values(by=['id'])
        df_brand_names = df_brand_names.drop(['brand_length_in_query','brand_name_in_attr', 'brand_name_size','brand_name_query'], axis=1)

        df_brand_names['product_info'] = df_brand_names['search_term']+"\t" + \
            df_brand_names['product_title']+"\t"+df_brand_names['product_description']\
             
        x = df_brand_names['product_info'].map(lambda x: found_common_words(
            x.split('\t')[0], x.split('\t')[1], x.split('\t')[2], nlp, stemmer))
        y = x.to_frame()
        # Add all the features.
        df_brand_names['query_root_in_desc'] = y['product_info'].map(lambda x: x[0])
        df_brand_names['query_compound_in_desc'] = y['product_info'].map(
            lambda x: x[1])
        df_brand_names['query_other_in_desc'] = y['product_info'].map(lambda x: x[2])
        df_brand_names['total_query_root'] = y['product_info'].map(lambda x: x[3])
        df_brand_names['total_query_compound'] = y['product_info'].map(lambda x: x[4])
        df_brand_names['total_query_other'] = y['product_info'].map(lambda x: x[5])
        df_brand_names['query_root_also_root_in_desc'] = y['product_info'].map(
            lambda x: x[6])
        df_brand_names['missing_query_terms'] = y['product_info'].map(
            lambda x: x[7])

        # store the featuers in 'filtered_train.csv'
        pd.DataFrame({"id": df_brand_names['id'], "product_uid": df_brand_names['product_uid'],"missing_query_terms": df_brand_names['missing_query_terms'],
                     "query_root_in_desc": df_brand_names['query_root_in_desc'], "query_compound_in_desc": df_brand_names["query_compound_in_desc"],"query_other_in_desc": df_brand_names["query_other_in_desc"], 
                     "total_query_root": df_brand_names["total_query_root"],"total_query_compound": df_brand_names["total_query_compound"],"total_query_other": df_brand_names["total_query_other"],
                     "query_root_also_root_in_desc": df_brand_names["query_root_also_root_in_desc"]}).to_csv('filtered_train_desc.csv', index=False)
        
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


    def generateSimilarityTitleDesc(self):
        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")
        df_pro_desc = pd.read_csv('./data/product_descriptions.csv', encoding="ISO-8859-1")
        nlp = spacy.load("en_core_web_lg")
        # nltk.download('wordnet') < needed if not installed yet (pip)
        nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
        nlp.add_pipe("spacy_wordnet", after='tagger')

        def calcSimilarity(id,sentence1,sentence2,nlp):
            doc1 = nlp(sentence1.lower())
            doc2 = nlp(sentence2.lower())
            print(id)
            return doc1.similarity(doc2)

        df_train = pd.merge(df_train, df_pro_desc, how='left', on='product_uid')

        df_train['query_title_similarity'] = df_train.apply(lambda row: calcSimilarity(row['id'],row['search_term'],row['product_title'],nlp), axis=1)
        df_train['query_desc_similarity'] = df_train.apply(lambda row: calcSimilarity(row['id'],row['search_term'],row['product_description'],nlp), axis=1)

        pd.DataFrame({"id": df_train['id'],"query_title_similarity":df_train['query_title_similarity'],"query_desc_similarity":df_train['query_desc_similarity']}).to_csv('filtered_title_desc_similarity.csv', index=False)


class scanData:
    def __init__(self):
        pass

    def analyzingFeatures(self):
        df_brand = pd.read_csv('./filtered_data/filtered_brand_name.csv',encoding="ISO-8859-1").fillna("")
        df_brand = df_brand[(df_brand['brand_length_in_query'] == df_brand['brand_name_size']) & (df_brand['brand_name_size'] > 0 )]
        df_features_title = pd.read_csv(
            './filtered_data/filtered_train_title.csv', encoding="ISO-8859-1")
        df_features_desc = pd.read_csv(
            './filtered_data/filtered_train_desc.csv', encoding="ISO-8859-1")
        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")

        merged_brand = pd.merge(df_train, df_brand, how='left', on=['id', 'product_uid'])
        merged_title = pd.merge(df_train, df_features_title, how='left', on=['id', 'product_uid'])
        merged_desc = pd.merge(df_train, df_features_desc, how='left', on=['id', 'product_uid'])


        # --- brand matching feature
        merged_brand = merged_brand.fillna(0)
        match_brand_df = merged_brand[merged_brand['brand_name_size'] > 0] # mean 2.487238 , std 0.472758
        no_match_brand_df = merged_brand[merged_brand['brand_name_size'] == 0] # mean 2.370361, std 0.538892
        # print("match with brand relevance")
        # print(match_brand_df['relevance'].describe())
        # print("no match with brand relevance")
        # print(no_match_brand_df['relevance'].describe())

        # --- title matching
        for feature in list(merged_title.columns)[6:]:
            print(feature, "True")
            print(merged_title[merged_title[feature] == 1]['relevance'].describe())
            print()
            print(feature, "False")
            print(merged_title[merged_title[feature] == 0]['relevance'].describe())

        """
                                                mean            std
        query_root_in_title [True]:             2.488044        0.485662
        query_root_in_title [False]:            2.206507        0.562980

        query_compound_in_title [True]:         2.427463        0.513000
        query_compound_in_title [False]:        2.331662        0.553225

        query_other_in_title [True]:            2.398902        0.528100
        query_other_in_title [False]:           2.360680        0.542896

        query_root_also_root_in_title[True]:    2.568459        0.446608
        query_root_also_root_in_title[False]:   2.334098        0.543877
        """

        print('\n\n\n\n\n\n')

        # --- Desc matching
        for feature in list(merged_desc.columns)[6:]:
            print(feature, "True")
            print(merged_desc[merged_desc[feature] == 1]['relevance'].describe())
            print()
            print(feature, "False")
            print(merged_desc[merged_desc[feature] == 0]['relevance'].describe())

        """
        query_root_in_desc [True]:              2.469301        0.495534
        query_root_in_desc [False]:             2.225250        0.563557

        query_compound_in_desc [True]:          2.424696        0.514064
        query_compound_in_desc [False]:         2.336462        0.554673

        query_other_in_desc [True]:             2.395592        0.526754   
        query_other_in_desc [False]:            2.372899        0.542705

        query_root_also_root_in_desc [True]:    2.465642        0.485094
        query_root_also_root_in_desc [False]:   2.380614        0.534472
        """



Features().generate_brand_matching_feature()

Features().generate_title_query_match()

Features().generate_desc_query_match()

Features().generateSimilarityTitleDesc()

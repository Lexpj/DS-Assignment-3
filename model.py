import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from keras.models import Sequential, load_model
from keras.layers import Dense

import sys

np.set_printoptions(threshold=sys.maxsize)
import shap
import xgboost

# from nltk.corpus import wordnet
# import contextualSpellCheck
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        time1 = time.perf_counter()
        func(*args, **kwargs)
        time2 = time.perf_counter()
        print(f"Took {time2-time1} seconds!")
    return wrapper


class ML:

    def __init__(self,similarity=False):
        data = self.__makeMergedDataset(similarity=similarity)
        self.X = data[0]
        self.y = data[1]

    def __makeMergedDataset(self):
        df_filtered_data = pd.read_csv('./filtered_data/filtered_train.csv',encoding="ISO-8859-1")
        df_filtered_attr = pd.read_csv('./filtered_data/filtered_attr.csv',encoding="ISO-8859-1")
        df_data = pd.read_csv('./data/train.csv',encoding="ISO-8859-1")
        df_brand_names = pd.read_csv('./filtered_data/filtered_brand_name.csv',encoding="ISO-8859-1")
        df_filtered_default = pd.read_csv('./filtered_data/filtered_default.csv',encoding="ISO-8859-1")

        # df_filtered_data['word_in_title'] = df_filtered_default['word_in_title']
        # df_filtered_data['word_in_description'] = df_filtered_default['word_in_description']
        # df_filtered_data['len_of_query'] = df_filtered_default['len_of_query']
        
        # df_filtered_data['query_root_in_attr'] = df_filtered_attr['query_root_in_attr']
        # df_filtered_data['query_compound_in_attr'] = df_filtered_attr['query_compound_in_attr']
        # df_filtered_data['query_other_in_attr'] = df_filtered_attr['query_other_in_attr']

        # df_filtered_data['brand_length_in_query'] = df_brand_names['brand_length_in_query']
        # df_filtered_data['brand_name_size'] = df_brand_names['brand_name_size']

        df_filtered_data['brand_match'] = (df_brand_names['brand_length_in_query'] == df_brand_names['brand_name_size']).astype(int)

        relevanceScore = pd.merge(df_filtered_data,df_data, how='left', on='id')
        relevanceScore = relevanceScore['relevance'].values

        # df_filtered_data = df_filtered_data.drop(['len_of_query'], axis=1)
        # df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
        # df_filtered_data = df_filtered_data.drop([ 
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
        #                                         'brand_in_query'
        #                                         ], axis=1) # For time purposes, dropping irrelevant columns barely differs error, but differs significantly in time
        
        df_filtered_data = df_filtered_data.fillna(0)

        
        df_filtered_data = df_filtered_data.drop([
            'missing_query_terms'
        ], axis=1)

        print(df_filtered_data.columns)

        return df_filtered_data, relevanceScore

    def __makeMergedDataset(self,similarity):
        df_brand = pd.read_csv('./filtered_data/filtered_brand_name.csv',encoding="ISO-8859-1").fillna("")
        df_brand['brand_match'] = ((df_brand['brand_length_in_query'] == df_brand['brand_name_size']) & (df_brand['brand_name_size'] > 0 )).astype(int)

        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")
        
        df_features_title = pd.read_csv(
            './filtered_data/filtered_train_title.csv', encoding="ISO-8859-1")
        df_features_desc = pd.read_csv(
            './filtered_data/filtered_train_desc.csv', encoding="ISO-8859-1")


        merged = pd.merge(df_train, df_brand, how='left', on=['id', 'product_uid'])
        merged = pd.merge(merged, df_features_title, how='left', on=['id', 'product_uid'])
        merged = pd.merge(merged, df_features_desc, how='left', on=['id', 'product_uid'])

        if similarity:
            df_sim = pd.read_csv('./filtered_data/filtered_title_desc_similarity.csv', encoding="ISO-8859-1")
            merged = pd.merge(merged, df_sim, how='left', on=['id'])

        merged = merged.fillna(0)
        relevanceScore = merged['relevance'].values
        merged['total_query_root'] = merged['total_query_root_y']
        merged['total_query_compound'] = merged['total_query_compound_y']
        merged['total_query_other'] = merged['total_query_other_y']

        merged = merged.drop([
            'relevance',
            'product_title',
            'search_term',
            'brand_name_query',
            'brand_name_in_attr',
            'missing_query_terms_x',
            'missing_query_terms_y',
            'brand_length_in_query',
            'brand_name_size',
            'total_query_compound_x',
            'total_query_root_x',
            'total_query_other_x',
            'total_query_compound_y',
            'total_query_root_y',
            'total_query_other_y'
        ], axis=1)

        print(merged.columns)
        merged.to_csv("./filtered_data/test.csv")
        self.nr_features = len(merged.columns)-2

        return merged, relevanceScore


        
    def __get_train_test_split(self,X,y):
        X = X.drop(['id', 'product_uid'], axis=1).values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, train_size=0.8)
        
        return X_train, X_test, y_train, y_test
    

    
    def eval_model(self,model):
        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.X,self.y) 

        model.fit(X_train, y_train)

        shapper = Shapper(model, 13)
        shapper(X_test, X_test, y_test)
    
    def eval_importances(self,model):
        l = [1.0, 1.25, 1.33, 1.5, 1.67, 1.75, 2.0, 2.25, 2.33, 2.5, 2.67, 2.75, 3.0]
        def mapToOnehot(y):
            item = np.zeros(len(l))
            item[l.index(y)] = 1
            return np.array(item)
        def mapToValue(y):
            return l[np.argmax(np.array(y))]
        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.X,self.y) 

        if type(model) == RandomForestClassifier:
            y_train = np.array(list(map(mapToOnehot, y_train)))
            y_test = np.array(list(map(mapToOnehot, y_test)))

        importance = permutation_importance(model, np.concatenate((X_train,X_test),axis=0), np.concatenate((y_train,y_test),axis=0), n_repeats=1, random_state=0)
        print("feature importances",importance)

        df = self.X
        df = df.drop(['id', 'product_uid'], axis=1)

        plt.bar(df.columns,importance['importances_mean'],yerr=importance['importances_std'])
        plt.xticks(rotation='vertical')
        plt.show()

    @timeit
    def randomforest_regressor(self):
        l = [1.0, 1.25, 1.33, 1.5, 1.67, 1.75, 2.0, 2.25, 2.33, 2.5, 2.67, 2.75, 3.0]
        def mapToOneHot(y):
            s = np.zeros(len(l))
            s[l.index(y)] = 1
            return s
        def mapToValue(y):
            return l[np.argmax(np.array(y))]

        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.X,self.y) 

        # y_train = np.array(list(map(mapToOneHot,y_train)))
            
        param_grid = {
            'n_estimators': [100,200,300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2,5,10],
            'min_samples_leaf': [1,2,4],
            'max_features': ['sqrt','log2'],
            'bootstrap': [True,False]
        }


        hyperparameters = self.hyperparameter_optimization(RandomForestRegressor(), param_grid)
        # Optimal hyperparameters are 
        # {'bootstrap': True, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}
        
        model = RandomForestRegressor(**hyperparameters)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # y_pred = np.array(list(map(mapToValue,y_pred)))
        
        print(mean_squared_error(y_test,y_pred)**0.5) # Root mean squared error of 0.4866

    @timeit
    def gradientbooster_regressor(self):
        l = [1.0, 1.25, 1.33, 1.5, 1.67, 1.75, 2.0, 2.25, 2.33, 2.5, 2.67, 2.75, 3.0]
        def mapToOneHot(y):
            s = np.zeros(len(l))
            s[l.index(y)] = 1
            return s
        def mapToValue(y):
            return l[np.argmax(np.array(y))]

        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.X,self.y) 

        # y_train = np.array(list(map(mapToOneHot,y_train)))
            
        param_grid = {
            'n_estimators': [300],#[100,200,300],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.01],#,0.1,1],
            'min_samples_split': [2,5,10],
            'min_samples_leaf': [1,2,4],
            'max_features': ['log2']#['sqrt','log2'],
        }


        hyperparameters = self.hyperparameter_optimization(GradientBoostingRegressor(), param_grid)
        # Optimal hyperparameters are 
        # {'learning_rate': 0.1, 'max_depth': 5, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 300}
        
        model = GradientBoostingRegressor(**hyperparameters)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # y_pred = np.array(list(map(mapToValue,y_pred)))
        
        print(mean_squared_error(y_test,y_pred)**0.5) # Root mean squared error of 0.49285043172056214

    @timeit
    def histgradientbooster_regressor(self):
        l = [1.0, 1.25, 1.33, 1.5, 1.67, 1.75, 2.0, 2.25, 2.33, 2.5, 2.67, 2.75, 3.0]
        def mapToOneHot(y):
            s = np.zeros(len(l))
            s[l.index(y)] = 1
            return s
        def mapToValue(y):
            return l[np.argmax(np.array(y))]

        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.X,self.y) 

        # y_train = np.array(list(map(mapToOneHot,y_train)))
            
        param_grid = {
            'learning_rate': [0.01,0.1,1],
            'min_samples_leaf': [2,5,10],
            'max_bins': [self.nr_features],
            'max_depth': [3,5,10],
            'l2_regularization': [0.0,1.0,2.0]
        }


        hyperparameters = self.hyperparameter_optimization(HistGradientBoostingRegressor(), param_grid)
        # Optimal hyperparameters are 
        # {'l2_regularization': 0.0, 'learning_rate': 0.1, 'max_bins': 13, 'max_depth': 10, 'min_samples_leaf': 10}
        
        model = HistGradientBoostingRegressor(**hyperparameters)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # y_pred = np.array(list(map(mapToValue,y_pred)))
        
        print(mean_squared_error(y_test,y_pred)**0.5) # Root mean squared error of 0.48641829810279763

    @timeit
    def neuralnet_regressor(self):
        l = [1.0, 1.25, 1.33, 1.5, 1.67, 1.75, 2.0, 2.25, 2.33, 2.5, 2.67, 2.75, 3.0]
        def mapToOnehot(y):
            item = np.zeros(len(l))
            item[l.index(y)] = 1
            return np.array(item)
        def mapToValue(y):

            return l[np.argmax(np.array(y))]

        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.X,self.y) 

        y_train = np.array(list(map(mapToOnehot, y_train)))
            
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(len(X_train[0]),)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(l), activation='softmax'))  

        model.compile(loss='mse',optimizer='adam', metrics=['accuracy','mse'])

        model.fit(X_train, y_train, batch_size=1, epochs=10)

        y_pred = model.predict(X_test)

        print(y_pred)

        y_pred = np.array(list(map(mapToValue, y_pred)))
        
        

        print(mean_squared_error(y_test,y_pred)**0.5) # Optimally 0.58
        
        model.save("./neuralmodel")

    @timeit
    def randomforest_classifier(self):
        l = [1.0, 1.25, 1.33, 1.5, 1.67, 1.75, 2.0, 2.25, 2.33, 2.5, 2.67, 2.75, 3.0]
        def mapToOnehot(y):
            item = np.zeros(len(l))
            item[l.index(y)] = 1
            return np.array(item)
        def mapToValue(y):
            return l[np.argmax(np.array(y))]

        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.X,self.y) 

        y_train = np.array(list(map(mapToOnehot, y_train)))

        param_grid = {
            'min_samples_leaf': [1,2,4],
            'min_samples_split': [2,5,10],
            'max_depth': [None,5,10],
            'n_estimators': [100,200,300]
        }

        hyperparameters = self.hyperparameter_optimization(RandomForestClassifier(), param_grid, oneHot=True)

        # Optimal hyperparameters are 
        # {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}
        
        model = RandomForestClassifier(**hyperparameters)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = np.array(list(map(mapToValue,y_pred)))
        
        print(mean_squared_error(y_test,y_pred)**0.5) # Root mean squared error of 1.396105225811824

    def hyperparameter_optimization(self, model, param_grid, oneHot=False):
        """
        This function is used to GridSearch the optimal hyperparameters for HistGradientBoostingRegressor
        """
        l = [1.0, 1.25, 1.33, 1.5, 1.67, 1.75, 2.0, 2.25, 2.33, 2.5, 2.67, 2.75, 3.0]
        def mapToOnehot(y):
            item = np.zeros(len(l))
            item[l.index(y)] = 1
            return np.array(item)
        
        X_train, _, y_train, _ = self.__get_train_test_split(self.X,self.y)        # <- self.getSplits()

        if oneHot:
            y_train = np.array(list(map(mapToOnehot, y_train)))
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error',verbose=2)

        #grid_search = RandomizedSearchCV(HistGradientBoostingRegressor(), param_grid, cv=10)
        grid_search.fit(X_train, y_train)


        print(grid_search.best_params_)
        print(grid_search.best_score_)

        with open("results.txt","a+") as f:
            f.write(str(type(model)) + " " + str(grid_search.best_params_) + "\n")

        return grid_search.best_params_


    def try_model(self, model, n_rep = 10, explain=False):
        """
        Test the model with some training data to see what the error is
        """
        sum_error = 0
        sum_time = 0

        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.X,self.y) 

        for i in range(n_rep):
            time1 = time.perf_counter()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            time2 = time.perf_counter()
            error = mean_squared_error(y_test, y_pred, squared=False)
            
            sum_error += error
            sum_time += time2-time1

        print("Performance of model:",type(model),f"n_rep={n_rep}")
        print("Average error:",sum_error/n_rep, "Average time", sum_time/n_rep)

        return sum_error/n_rep, sum_time/n_rep

class Model:

    def __init__(self):
        data = self.__makeMergedDataset()
        self.X = data[0]
        self.y = data[1]
        datasim = self.__makeMergedDatasetSim()
        self.Xsim = datasim[0]
        self.ysim = datasim[1]

    def __makeMergedDatasetSim(self):
        df_brand = pd.read_csv('./filtered_data/filtered_brand_name.csv',encoding="ISO-8859-1").fillna("")
        df_brand['brand_match'] = ((df_brand['brand_length_in_query'] == df_brand['brand_name_size']) & (df_brand['brand_name_size'] > 0 )).astype(int)

        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")
        
        df_features_title = pd.read_csv(
            './filtered_data/filtered_train_title.csv', encoding="ISO-8859-1")
        df_features_desc = pd.read_csv(
            './filtered_data/filtered_train_desc.csv', encoding="ISO-8859-1")


        merged = pd.merge(df_train, df_brand, how='left', on=['id', 'product_uid'])
        merged = pd.merge(merged, df_features_title, how='left', on=['id', 'product_uid'])
        merged = pd.merge(merged, df_features_desc, how='left', on=['id', 'product_uid'])

        df_sim = pd.read_csv('./filtered_data/filtered_title_desc_similarity.csv', encoding="ISO-8859-1")
        merged = pd.merge(merged, df_sim, how='left', on=['id'])

        merged = merged.fillna(0)
        relevanceScore = merged['relevance'].values
        merged['total_query_root'] = merged['total_query_root_y']
        merged['total_query_compound'] = merged['total_query_compound_y']
        merged['total_query_other'] = merged['total_query_other_y']

        merged = merged.drop([
            'relevance',
            'product_title',
            'search_term',
            'brand_name_query',
            'brand_name_in_attr',
            'missing_query_terms_x',
            'missing_query_terms_y',
            'brand_length_in_query',
            'brand_name_size',
            'total_query_compound_x',
            'total_query_root_x',
            'total_query_other_x',
            'total_query_compound_y',
            'total_query_root_y',
            'total_query_other_y'
        ], axis=1)

        print(merged.columns)
        merged.to_csv("./filtered_data/test.csv")
        self.nr_features = len(merged.columns)-2

        return merged, relevanceScore

        return df_filtered_data, relevanceScore

    def __makeMergedDataset(self):
        df_brand = pd.read_csv('./filtered_data/filtered_brand_name.csv',encoding="ISO-8859-1").fillna("")
        df_brand['brand_match'] = ((df_brand['brand_length_in_query'] == df_brand['brand_name_size']) & (df_brand['brand_name_size'] > 0 )).astype(int)

        df_train = pd.read_csv(
            './data/train.csv', encoding="ISO-8859-1")
        
        df_features_title = pd.read_csv(
            './filtered_data/filtered_train_title.csv', encoding="ISO-8859-1")
        df_features_desc = pd.read_csv(
            './filtered_data/filtered_train_desc.csv', encoding="ISO-8859-1")


        merged = pd.merge(df_train, df_brand, how='left', on=['id', 'product_uid'])
        merged = pd.merge(merged, df_features_title, how='left', on=['id', 'product_uid'])
        merged = pd.merge(merged, df_features_desc, how='left', on=['id', 'product_uid'])


        merged = merged.fillna(0)
        relevanceScore = merged['relevance'].values
        
        merged = merged.drop([
            'relevance',
            'product_title',
            'search_term',
            'brand_name_query',
            'brand_name_in_attr',
            'missing_query_terms_x',
            'missing_query_terms_y',
            'brand_length_in_query',
            'brand_name_size',
            'total_query_compound_x',
            'total_query_root_x',
            'total_query_other_x'
        ], axis=1)

        print(merged.columns)
        merged.to_csv("./filtered_data/test.csv")
        self.nr_features = len(merged.columns)-2

        return merged, relevanceScore
    
    def __get_train_test_split(self,X,y):
        X = X.drop(['id', 'product_uid'], axis=1).values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, train_size=0.8)
        
        return X_train, X_test, y_train, y_test
    
    def randomforest_regressor(self):
        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.X,self.y) 

        hyperparameters = {'bootstrap': True, 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 200}

        model = RandomForestRegressor(**hyperparameters)
        model.fit(X_train, y_train)
        return model
    
    def gradientbooster_regressor(self):
        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.X,self.y) 

        hyperparameters = {'learning_rate': 0.01, 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 300}
        
        model = GradientBoostingRegressor(**hyperparameters)
        model.fit(X_train, y_train)
        return model

    def histgradientbooster_regressor(self):
        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.X,self.y) 

        hyperparameters = {'l2_regularization': 1.0, 'learning_rate': 0.1, 'max_bins': 12, 'max_depth': 10, 'min_samples_leaf': 10}

        model = HistGradientBoostingRegressor(**hyperparameters)
        model.fit(X_train, y_train)

        return model

    def neuralnet_regressor(self):
        model = load_model("./neuralmodel")
        return model

    def randomforest_classifier(self):
        l = [1.0, 1.25, 1.33, 1.5, 1.67, 1.75, 2.0, 2.25, 2.33, 2.5, 2.67, 2.75, 3.0]
        def mapToOnehot(y):
            item = np.zeros(len(l))
            item[l.index(y)] = 1
            return np.array(item)
        def mapToValue(y):
            return l[np.argmax(np.array(y))]

        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.X,self.y) 

        y_train = np.array(list(map(mapToOnehot, y_train)))

        hyperparameters = {'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}

        # Optimal hyperparameters are 
        # 
        
        model = RandomForestClassifier(**hyperparameters)
        model.fit(X_train, y_train)

        return model
    
    def randomforest_regressor_sim(self):
        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.Xsim,self.ysim) 

        hyperparameters = {'bootstrap': True, 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 300}

        model = RandomForestRegressor(**hyperparameters)
        model.fit(X_train, y_train)
        return model
    
    def gradientbooster_regressor_sim(self):
        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.Xsim,self.ysim) 

        hyperparameters = {'learning_rate': 0.01, 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 300}

        
        model = GradientBoostingRegressor(**hyperparameters)
        model.fit(X_train, y_train)
        return model

    def histgradientbooster_regressor_sim(self):
        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.Xsim,self.ysim) 

        hyperparameters = {'l2_regularization': 0.0, 'learning_rate': 0.1, 'max_bins': 14, 'max_depth': 10, 'min_samples_leaf': 5}


        model = HistGradientBoostingRegressor(**hyperparameters)
        model.fit(X_train, y_train)

        return model

    def neuralnet_regressor_sim(self):
        model = load_model("./neuralmodel")
        return model

    def randomforest_classifier_sim(self):
        l = [1.0, 1.25, 1.33, 1.5, 1.67, 1.75, 2.0, 2.25, 2.33, 2.5, 2.67, 2.75, 3.0]
        def mapToOnehot(y):
            item = np.zeros(len(l))
            item[l.index(y)] = 1
            return np.array(item)
        def mapToValue(y):
            return l[np.argmax(np.array(y))]

        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.Xsim,self.ysim) 

        y_train = np.array(list(map(mapToOnehot, y_train)))

        hyperparameters = {'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 300}


        # Optimal hyperparameters are 
        # 
        
        model = RandomForestClassifier(**hyperparameters)
        model.fit(X_train, y_train)

        return model
    
    def shap_data(self):
        X_train, X_test, y_train, y_test = self.__get_train_test_split(self.X,self.y) 
        return X_train, X_test, y_test

class Shapper:

    def __init__(self, model, nr_features):
        self.model = model
        self.nr_features = nr_features
        self.amount = 1000

    def explain(self, X_train, X_test, y_test, *args):
        self.explainer = shap.Explainer(self.model, X_train[:self.amount])#, feature_names=self.nr_features)
        self.shap_values = self.explainer(X_test[:self.amount])

    def show(self, X_train, X_test, y_test, *args):
        print("Actual:",float(y_test[0]))
        shap.plots.waterfall(self.shap_values[0])
        shap.plots.beeswarm(self.shap_values)
        # Add the actual target value as an annotation
        
        plt.show()

    def __call__(self, *args):
        self.explain(*args)
        self.show(*args)


def plot_importances(models):
    pass


# weight sharing between states. slide 21
# slide 25: group states, estimate one value for each group/buckets
# slide 26: linear methods: define feature vector with basic functions, and the value for a state is the inproduct of a combinatino of w,s
# 2D plynormial features slide 29



x = ML(similarity=True)
# x.eval_model(RandomForestRegressor(
#     **{'bootstrap': True, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}
# ))
# x.eval_importances(RandomForestRegressor(
#     **{'bootstrap': True, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}
# ))
# x.neuralnet_regressor()
# x.randomforest_regressor()
# x.gradientbooster_regressor()
# x.histgradientbooster_regressor()
# x.randomforest_classifier()

modelclass = Model()

cur = modelclass.randomforest_classifier_sim()
print("gotmodel")
x.eval_importances(cur)



model = modelclass.randomforest_regressor()
s = Shapper(model, modelclass.nr_features)
s(*modelclass.shap_data())
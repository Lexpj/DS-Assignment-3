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
# import contextualSpellCheck
import time

class Week7:

    def __init__(self):
        data = self.makeMergedDataset()
        self.X = data[0]
        self.y = data[1]


    def regression_models(self):
        X_train, X_test, y_train, y_test = self.get_train_test_split(self.X,self.y)     # <- self.getSplits()

        # Naked models (all without bagging)
        # self.try_model(X_train, y_train, X_test, y_test, RandomForestRegressor())              # 0.559, t=21.99
        # self.try_model(X_train, y_train, X_test, y_test, KNeighborsRegressor())                # 0.570, t=0.11
        # self.try_model(X_train, y_train, X_test, y_test, HistGradientBoostingRegressor())      # 0.481, t=0.29
        # self.try_model(X_train, y_train, X_test, y_test, SVR())                                # 0.532, t=179
        
        self.try_model(X_train, y_train, X_test, y_test, HistGradientBoostingRegressor(
            **{'learning_rate': 0.12, 'max_depth': 15, 'max_iter': 400, 'min_samples_leaf': 17}
        ),n_rep=25) #0.478, t=0.54
        

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

        df = self.X
        df = df.drop(['id', 'product_uid'], axis=1)

        plt.bar(df.columns,importance['importances_mean'],yerr=importance['importances_std'])
        plt.xticks(rotation='vertical')
        plt.show()

        print("Performance of model:",type(model),f"n_rep={n_rep}")
        print("Average error:",sum_error/n_rep, "Average time", sum_time/n_rep)

        return sum_error/n_rep, sum_time/n_rep

    def makeMergedDataset(self):
        df_filtered_data = pd.read_csv('./filtered_data/filtered_train.csv',encoding="ISO-8859-1")
        df_filtered_attr = pd.read_csv('./filtered_data/filtered_attr.csv',encoding="ISO-8859-1")
        df_data = pd.read_csv('./data/train.csv',encoding="ISO-8859-1")
        df_brand_names = pd.read_csv('./filtered_data/filtered_brand_name.csv',encoding="ISO-8859-1")
        df_filtered_default = pd.read_csv('./filtered_data/filtered_default.csv',encoding="ISO-8859-1")

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
        #                                         'brand_in_query'], axis=1) # For time purposes, dropping irrelevant columns barely differs error, but differs significantly in time
        
        return df_filtered_data, relevanceScore

    def get_train_test_split(self,X,y):
        X = X.drop(['id', 'product_uid'], axis=1).values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, train_size=0.8)
        
        return X_train, X_test, y_train, y_test
    

x = Week7()
x.regression_models()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler

class Feature_and_Splitting():
    def __init__(self, X, y, test_size):
        """
        Parameters:

        X: Features, the values used to learn from

        y: The value being predicted
        
        test_size: Amount of data wanted to not be trained on 
        and kept for testing it is a value between 0 to 1.
        """

        self.X = X
        self.y = y
        self.test_size = test_size
        
    def train_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = 0)
        return X_train, X_test, y_train, y_test
    
    def get_features(self, amount, model, fit = 'Test'):
        X_train, X_test, y_train, y_test = self.train_split()
        
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        
        X_train_sc = sc_X.fit_transform(X_train)
        X_test_sc = sc_X.transform(X_test)

        if model == 'Regression':
            try:
                y_train_sc = sc_y.fit_transform(np.array(y_train).reshape(-1, 1))
                y_test_sc = sc_y.transform(np.array(y_test).reshape(-1, 1))
            except AttributeError:
                y_train_sc = sc_y.fit_transform(y_train)
                y_test_sc = sc_y.transform(y_test)

            if fit == 'Test':
                self.features = X_test_sc
                self.target = y_test_sc

            else:
                self.features = X_train_sc
                self.target = y_train_sc

                
            # self.select = SelectPercentile(percentile = amount)
            self.select = SelectKBest(k = amount)
            self.select.fit(self.features, self.target)
            X_train_selected = self.select.transform(X_train_sc)
            X_test_selected = self.select.transform(X_test_sc)
                
            return X_train_selected, X_test_selected, y_train_sc, y_test_sc, sc_X, sc_y, self.select, y_test
        
        if model == 'Classification':

            if fit == 'Test':
                self.features = X_test_sc
                self.target = y_test

            else:
                self.features = X_train_sc
                self.target = y_train

            # self.select = SelectPercentile(percentile = amount)
            self.select = SelectKBest(k = amount)
            self.select.fit(self.features, self.target)
            X_train_selected = self.select.transform(X_train_sc)
            X_test_selected = self.select.transform(X_test_sc)
        
            return X_train_selected, X_test_selected, sc_X, sc_y, self.select
    
    def show_kept_features(self, all_features):  # Calls self.train_split() TWICE
        
        mask = self.select.get_support()
        new_features = []
        dropped = []
        
        for bool, feature in zip(mask, all_features):
            if bool:
                new_features.append(feature)
            else:
                dropped.append(feature)
        
        plt.figure(figsize = (10, 7))
        plt.matshow(mask.reshape(1, -1), cmap = 'gray_r')
        print("Selected Features:")
        plt.show()
        
        return mask, new_features, dropped
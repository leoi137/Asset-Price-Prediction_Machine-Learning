import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

class RandomForestAlgorithm(object):
    def __init__(self, X_train_selected, X_test_selected, y_train, y_test):

        """
        Parameters:

        X_train_selected: The selected features  for train on (called selected because it assumes
        they have gone through a feature selection process, works either way.)

        X_test_selected: The selected featues to test on

        y_train: The value being predicted for training

        y_test: The value being predicted for testing
        """

        self.X_train = X_train_selected
        self.X_test = X_test_selected
        self.y_train = y_train
        self.y_test = y_test

    def random_forest_classifier(self, n_jobs, estimators, step_factor, axis_step):

        """
        Parameters:

        estimators: The amount of trees wanted to be used.

        step_factor: The steps wanted to take to measure each tree, for
        example if its 10 and you choose 30 trees, you will test on every 10
        tress until reaching 30

        axis_step: The amount of steps taken (estimators divided by step_factor)

        n_jobs = Number of cores wanting to use from your CPU, -1 
        means all cores available.

        """

        self.n_jobs = n_jobs
        self.n_estimators = estimators
        self.step_factor = step_factor
        self.axis_step = axis_step
        
        self.estimators = np.zeros(self.axis_step)
        self.rf_acc_train = np.zeros(self.axis_step)
        self.rf_acc_test = np.zeros(self.axis_step)

        for i in range(0, self.axis_step):
            print("Random Forest Estimator: {0} of {1}".format(
                self.step_factor * (i + 1), self.n_estimators
            )
                 )
            self.rf = RandomForestClassifier(
                n_estimators = self.step_factor*(i +1),
                n_jobs = self.n_jobs, 
                random_state = 0
            )
            self.rf.fit(self.X_train, self.y_train)
            self.train_predictions = self.rf.predict(self.X_train)
            self.test_predictions = self.rf.predict(self.X_test)
            self.acc_train = accuracy_score(self.y_train, self.train_predictions, normalize = True)
            self.acc_test = accuracy_score(self.y_test, self.test_predictions, normalize = True)
            self.estimators[i] = self.step_factor*(i + 1)
            self.rf_acc_train[i] = self.acc_train
            self.rf_acc_test[i] = self.acc_test
            
        return self.rf_acc_train, self.rf_acc_test, self.estimators, self.rf

    def random_forest_regressor(self, n_jobs, estimators, step_factor, axis_step):

        """
        Parameters:

        estimators: The amount of trees wanted to be used.

        step_factor: The steps wanted to take to measure each tree, for
        example if its 10 and you choose 30 trees, you will test on every 10
        tress until reaching 30

        axis_step: The amount of steps taken (estimators divided by step_factor)

        n_jobs = Number of cores wanting to use from your CPU, -1 
        means all cores available.

        """
        
        self.n_jobs = n_jobs
        self.n_estimators = estimators
        self.step_factor = step_factor
        self.axis_step = axis_step
        
        self.estimators = np.zeros(self.axis_step)
        self.rf_acc_train = np.zeros(self.axis_step)
        self.rf_acc_test = np.zeros(self.axis_step)

        for i in range(0, self.axis_step):
            print("Random Forest Estimator: {0} of {1}".format(
                self.step_factor * (i + 1), self.n_estimators
            )
                 )
            self.rf = RandomForestRegressor(
                n_estimators = self.step_factor*(i +1),
                n_jobs = self.n_jobs, 
                random_state = 0
            )
            self.rf.fit(self.X_train, self.y_train)
            self.train_predictions = self.rf.predict(self.X_train)
            self.test_predictions = self.rf.predict(self.X_test)
            self.acc_train = mean_absolute_error(self.y_train, self.train_predictions)
            self.acc_test = mean_absolute_error(self.y_test, self.test_predictions)
            self.estimators[i] = self.step_factor*(i + 1)
            self.rf_acc_train[i] = self.acc_train
            self.rf_acc_test[i] = self.acc_test
            
        return self.rf_acc_train, self.rf_acc_test, self.estimators, self.rf

    def graph(self, train_accuracy, test_accuracy, estimators):
        plt.figure(figsize = (10, 8))
        plt.title("Random Forest")
        plt.plot(estimators, train_accuracy, color = 'orange', label = 'Train')
        plt.plot(estimators, test_accuracy, color = 'blue', label = 'Test')
        plt.legend()
        plt.xlabel("Estimators")
        plt.ylabel("Measurement")
        plt.show()
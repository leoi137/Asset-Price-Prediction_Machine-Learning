import numpy as np
import matplotlib.pyplot as plt


from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

class KNNAlgorithm:
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
        self.y_train = y_train
        self.X_test = X_test_selected
        self.y_test = y_test
        
    def KNN_Algorithm_Classifier(self, neighbors, metric, n_jobs):
        """
        Parameters:

        Neighbors: The k nearest neighbors to choose from in list format

        Metric: The metric used to to measure the distance from the
        neighbors such as eucledian

        n_jobs = Number of cores wanting to use from your CPU, -1 
        means all cores available.

        """
        KNN_train_accuracy = np.zeros(len(neighbors))
        KNN_test_accuracy = np.zeros(len(neighbors))
        KNN_neighbors = np.zeros(len(neighbors))
        
        print("K-Nearest Neightbors")
        print("-----------------------")
        for i, ind in zip(neighbors, range(0, len(neighbors))):
            print("Neighbor {0} of {1}".format(i, neighbors[-1]))
            KNN = KNeighborsClassifier(
                n_neighbors = i, metric = metric, n_jobs = n_jobs
            )
            KNN.fit(self.X_train, self.y_train)
            train_predictions = KNN.predict(self.X_train)
            test_predictions = KNN.predict(self.X_test)
            accuracy_train = accuracy_score(self.y_train, train_predictions, normalize = True)
            accuracy_test = accuracy_score(self.y_test, test_predictions, normalize = True)
            KNN_train_accuracy[ind] = accuracy_train
            KNN_test_accuracy[ind] = accuracy_test
            KNN_neighbors[ind] = i
        
        self.graph(KNN_train_accuracy, KNN_test_accuracy, KNN_neighbors)
        
        return KNN_train_accuracy, KNN_test_accuracy, KNN_neighbors

    def KNN_Algorithm_Regression(self, neighbors, metric, n_jobs):

        """
        Parameters:

        Neighbors: The k nearest neighbors to choose from in list format

        Metric: The metric used to to measure the distance from the
        neighbors such as eucledian

        n_jobs = Number of cores wanting to use from your CPU, -1 
        means all cores available.

        """
        
        KNN_train_accuracy = np.zeros(len(neighbors))
        KNN_test_accuracy = np.zeros(len(neighbors))
        KNN_neighbors = np.zeros(len(neighbors))
        
        print("K-Nearest Neightbors")
        print("-----------------------")
        for i, ind in zip(neighbors, range(0, len(neighbors))):
            print("Neighbor {0} of {1}".format(i, neighbors[-1]))
            KNN = KNeighborsRegressor(
                n_neighbors = i, metric = metric, n_jobs = n_jobs
            )
            KNN.fit(self.X_train, self.y_train)
            train_predictions = KNN.predict(self.X_train)
            test_predictions = KNN.predict(self.X_test)
            accuracy_train = mean_squared_error(self.y_train, train_predictions)
            accuracy_test = mean_squared_error(self.y_test, test_predictions)
            KNN_train_accuracy[ind] = accuracy_train
            KNN_test_accuracy[ind] = accuracy_test
            KNN_neighbors[ind] = i
        
        self.graph(KNN_train_accuracy, KNN_test_accuracy, KNN_neighbors)
        
        return KNN_train_accuracy, KNN_test_accuracy, KNN_neighbors
    
    def graph(self, KNN_train_accuracy, KNN_test_accuracy, KNN_neighbors):
        plt.figure(figsize = (10, 8))
        plt.title("K-Nearest Neighbors")
        plt.plot(KNN_neighbors, KNN_train_accuracy, color = 'orange', label = 'Train')
        plt.plot(KNN_neighbors, KNN_test_accuracy, color = 'blue', label = 'Test')
        plt.legend()
        plt.xlabel("Estimators")
        plt.ylabel("Mean Squared Error")
        plt.show()
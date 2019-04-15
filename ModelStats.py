import pandas as pd
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class Statistics:
    def __init__(self, X_test_selected, y_test, Model, model_features, scores, size = 3):
        """
        Parameters:
        
        THIS ONLY WORKS WITH A CLASSIFIER MODEL WITH 3 PREDICTED VALUES

        X_test_selected: The selected featues to test on (called selected because it assumes
        they have gone through a feature selection process, works either way.)

        y_test: The value being predicted for testing

        Model: The trained model

        model_features: The name of the features used

        scores: The results after running k-Fold cross validation

        size: Only supports 3, meaning three different predicted values

        """

        self.X_test_selected = X_test_selected
        self.y_test = y_test
        self.size = size
        self.Model = Model
        self.model_feat = model_features
        self.scores = scores
        
        self.data_dict = {}
        
    
    def statistics(self):
        y_preds = self.Model.predict(self.X_test_selected)
        conf_matrix = confusion_matrix(self.y_test, y_preds)
        conf_DF = pd.DataFrame(conf_matrix, columns = ['-1', '0', '1'], index = ['-1', '0', '1'])
        if self.size == 3:
            bull_mean = (conf_matrix[0, 0]/ (conf_matrix[0, 0] + conf_matrix[1, 0] + conf_matrix[2, 0]))
            bear_mean = (conf_matrix[2, 2]/ (conf_matrix[0, 2] + conf_matrix[1, 2] + conf_matrix[2, 2]))
            none_mean = (conf_matrix[1, 1]/ (conf_matrix[0, 1] + conf_matrix[1, 1] + conf_matrix[2, 1]))
        #             inf = [scores.mean(), bull_mean, bear_mean, none_mean]
        #             index = ['All', 'Bullish', 'Bearish', 'Stalled']
        #             stats = pd.DataFrame(inf, index = index)
        #             stats.columns = ['Accuracy (%)']
        #             self.data_dict['Confusion Matrix'] = conf_DF
        #             self.data_dict['Accuracy'] = stats
            
            self.data_dict['Accuracy'] = {'All': accuracy_score(y_preds, self.y_test), 
                                          'Bull': bull_mean, 
                                          'Bear': bear_mean, 
                                          'Stalled': none_mean, 
                                          'STDV': self.scores.std()}
            self.data_dict['Confusion Matrix'] = {'-1': (int(conf_matrix[0, 0]), int(conf_matrix[1, 0]), 
                                                         int(conf_matrix[2, 0])), 
                                                 '0': (int(conf_matrix[0, 1]), int(conf_matrix[1, 1]), 
                                                       int(conf_matrix[2, 1])), 
                                                 '1': (int(conf_matrix[0, 2]), int(conf_matrix[1, 2]), 
                                                       int(conf_matrix[2, 2]))}
            self.data_dict['Features'] = self.model_feat
            

            return self.data_dict
        
    def save_file(self, model_name):
        """
        Parameters:

        model_name: The name of a file to save it as in string format
        """
        stats = self.statistics()
        with open('{}.json'.format(model_name), 'w') as outfile:
            json.dump(stats, outfile)
            
#         with open('DoublePriceClassifierDaily.csv', 'w') as f:
#             for key in stats.keys():
#                 f.write("%s,%s\n"%(key, stats[key]))

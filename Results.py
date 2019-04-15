import pandas as pd
from sklearn.metrics import confusion_matrix


class PredictionResults:
    def __init__(self, model, features, actual_value):

        """
        Parameters:

        model: The trained model

        features: The features used to predict (X values)

        actual_value: The True value being predicted (y value)

        """
        self.model = model
        self.features = features
        self.actual_value = actual_value
#         self.y_scaler = y_scaler
        
    def create_DataFrame(self, kind):
        """
        Parameters:
        
        kind: Classification or Regression

        """

        y_pred = self.model.predict(self.features)
        actual = self.actual_value
        preds = pd.DataFrame(y_pred, index = actual.index)
        results = pd.concat([actual, preds], axis = 1).sort_index()
        results.columns = ['Actual','Predictions']
        
        if kind == 'Classification':
            conf_matrix = confusion_matrix(actual, preds)
            print(conf_matrix)
            return results, conf_matrix
        elif kind == 'Regression':
            results['Error'] = results['Predictions'] - results['Actual']
            return results
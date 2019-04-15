from DataGathering import CSVData
from v3FeatureBuilder import FeatureData
from FeatureAndSplitting import Feature_and_Splitting
from Results import PredictionResults
from ModelStats import Statistics
from RandomForest import RandomForestAlgorithm

import time
import pandas as pd

csv_dir = 'C:\\Users\\leand\\Desktop\\Resources\\Data\\Forex\\Oanda\\Daily\\'
# symbols = ['eurusd', 'gbpusd', 'audusd', 'usdchf', 'usdjpy', 'eurjpy', 'eurgbp', 
#            'euraud', 'eurchf', 'audcad', 'audchf', 'cadchf', 'audjpy', 'cadjpy', 
#            'eurcad', 'usdcad', 'gbpaud', 'gbpchf', 'gbpcad', 'gbpjpy']

symbols = ['eurusd', 'gbpusd', 'audusd', 'usdchf', 'usdjpy', 'eurjpy', 'eurgbp']

if __name__ == '__main__':
	start_all = time.perf_counter()

	get_data = time.perf_counter()
	csv = CSVData(csv_dir, symbols, '2008', '2019', 'Forex')
	print("Gathering Data...\n")
	fx_data = csv.get_data()
	print("Gathering the data took: {:0.4f} seconds\n".format(time.perf_counter() - get_data))

	print("Structuring The Data...")
	print("-------------------------")
	structure_start = time.perf_counter()
	strc = FeatureData(fx_data, symbols, kind = 'Classification', features = True)
	data = strc.run(True)
	print("Structuring the data took: {:0.4f} seconds\n".format(time.perf_counter() - structure_start))

	y_vals = data['Target']
	y = y_vals.values
	X_vals = data.drop(['Target'], axis = 1)
	X = X_vals.values
	    
	print("Gathering all data took: {:0.4f} seconds\n".format(time.perf_counter() - start_all))
	print("Number of rows:", len(data))
	print("Number of columns:", len(X_vals.columns))
	total = len(data) * len(data.columns)
	print("Total Data Points: {}".format(total))

	print("Feature Selection...")
	print("-------------------------")
	start_all = time.perf_counter()
	FnS = Feature_and_Splitting(X, y, test_size = 0.20)
	X_train, X_test, y_train, y_test = FnS.train_split()
	X_train_selected, X_test_selected, scaler_X, scaler_y, selected = FnS.get_features(10, 'Classification', fit = 'Test')
	chosen, new_feat, dropped = FnS.show_kept_features(X_vals)
	print("Feature Selection took: {:0.4f} seconds\n".format(time.perf_counter() - start_all))
	print("Number of new features:", len(new_feat))
	rows = len(data)
	cols = len(new_feat)

	total = rows * cols
	print("Total Data Points: {}".format(total))


start = time.perf_counter()

rf = RandomForestAlgorithm(X_train_selected, X_test_selected, y_train, y_test)

n_jobs = -1
n_estimators = 100
step_factor = 30
axis_step = int(n_estimators/step_factor)

train_res, test_res, estimators, last_model = rf.random_forest_classifier(n_jobs, n_estimators, step_factor, axis_step)
rf.graph(train_res, test_res, estimators)

print("Took: {} seconds".format(time.perf_counter() - start))


dropped_index = []
for ind, e in zip(range(len(selected.get_support())), selected.get_support()):
	if e == False:
		dropped_index.append(ind)
        
def dropforFolds(X_data, dropped_index):
	X_df = pd.DataFrame(scaler_X.transform(X_data))
	X_all = X_df.drop(dropped_index, axis = 1)
	return X_all

X_all = dropforFolds(X_vals, dropped_index)

start = time.perf_counter()

scores = cross_val_score(Classifier, X_all, y, cv = 5)

print("k-Fold took: {} seconds".format(time.perf_counter() - start))

S = Statistics(X_test_selected, y_test, Classifier, new_feat, scores)
stats = S.statistics()

print(stats)
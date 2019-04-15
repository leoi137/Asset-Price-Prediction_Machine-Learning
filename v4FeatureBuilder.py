import pandas as pd
import numpy as np
import time

class FeatureData():

	def __init__(self, data, symbols, kind):
		"""
		Parameters:

		data: The data assumes open, high, low, close are columns

		symbols: All the symbols used

		kind: Regression of Classification
		"""

		self.data = data
		self.symbols = symbols
		self.kind = kind

		self.start_bars = 0
		self.prev_bars = 50
		self.period = 1

	def main(self, combine):
		"""
		Parameters:

		combine: boolean value, if True then it combines all data

		"""

		print("Creating Features...\n")
		start = time.perf_counter()

		self.bar_data = self.change_type(self.data)
		self.bar_data = self.create_bars(self.bar_data)
		self.bar_data = self.derivative(self.bar_data)
		self.bar_data = self.pin_bar(self.bar_data)
		self.bar_data = self.perc_change(self.bar_data)
		self.bar_data = self.standard_deviation(self.bar_data)
		self.bar_data = self.bar_differences(self.bar_data)
		self.bar_data = self.bar_perc(self.bar_data)
		
		print("Creating the features took: {:0.4f} seconds\n".format(time.perf_counter() - start))
		
		self.bar_data = self.create_target(self.bar_data)
		
		print("Cleaning and formating the data...")
		start = time.perf_counter()
		self.bar_data = self.clean_data(self.bar_data, combine)
		print("Clearning the data took: {:0.4f} seconds".format(time.perf_counter() - start))


		return self.bar_data#.dropna()

	def change_type(self, data):

		print("Changing data type from 64-bit to 32-bit...")

		start = time.perf_counter()

		for s in self.symbols:
			data[s] = data[s].astype('float32')

		print("Type conversion took: {:0.4f}\n".format(time.perf_counter() - start))	

		return data

	def create_bars(self, data):

		print("Creating Bars...")
		start = time.perf_counter()

		for s in self.symbols:
			for col in data[s].columns:
				for i in range(self.start_bars, self.prev_bars, self.period):
					data[s]["{0} {1}".format(col, str(i + self.period))] = data[s]['{}'.format(col)].diff(i + self.period)
				for i in range(self.start_bars, self.prev_bars, self.period):
					data[s]["{0} p - {1}".format(col, str(i + self.period))] = data[s]['{}'.format(col)].shift(i + self.period)

		print("Bar creation took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def perc_change(self, data):

		print("Taking percentage change...")
		start = time.perf_counter()

		for s in self.symbols:
			data[s]['pct'] = data[s]['close'].pct_change() * 100
			data[s]['pct fut'] = data[s]['close'].shift(-1).pct_change() * 100

		for s in self.symbols:
			for i in range(self.start_bars, self.prev_bars, self.period):
				data[s]["pct {}".format(str(i + self.period))] = data[s]['close'].pct_change(i + self.period)

		print("Percentage change took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def derivative(self, data):

		print("Taking Derivatives")
		start = time.perf_counter()

		for s in self.symbols:
			data[s]['open - derivative'] = np.gradient(data[s]['open'])
			data[s]['high - derivative'] = np.gradient(data[s]['high'])
			data[s]['low - derivative'] = np.gradient(data[s]['low'])
			data[s]['close - derivative'] = np.gradient(data[s]['close'])

			for i in range(self.start_bars, self.prev_bars, self.period):
				data[s]['open - derivative {}'.format(i + self.period)] = data[s]['open - derivative'].shift(i + self.period)
				data[s]['high - derivative {}'.format(i + self.period)] = data[s]['high - derivative'].shift(i + self.period)
				data[s]['low - derivative {}'.format(i + self.period)] = data[s]['low - derivative'].shift(i + self.period)
				data[s]['close - derivative {}'.format(i + self.period)] = data[s]['close - derivative'].shift(i + self.period)

		print("Derivatives took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def standard_deviation(self, data):

		print("Taking STDV...")
		start = time.perf_counter()

		for s in self.symbols:
			for i in range(self.start_bars + 1, self.prev_bars, self.period):
				data[s]["close STDV {}".format(str(i + self.period))] = data[s]['close'].shift(1).rolling(i + self.period).std()
				data[s]["high STDV {}".format(str(i + self.period))] = data[s]['high'].shift(1).rolling(i + self.period).std()
				data[s]["low STDV {}".format(str(i + self.period))] = data[s]['low'].shift(1).rolling(i + self.period).std()
				data[s]["open STDV {}".format(str(i + self.period))] = data[s]['open'].shift(1).rolling(i + self.period).std()

		print("STDV took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def pin_bar(self, data):

		print("Creating Pin bars...")
		start = time.perf_counter()

		# prev_pins = 10
		# pin_period = 1

		for s in self.symbols:

			data[s]['diff'] = (data[s]['high'] - data[s]['low']).abs()
			data[s]['std'] = data[s]['diff'].rolling(20).std()

			bull_bullcpin = (((data[s]['open'] - data[s]['low'])/(
				data[s]['high'] - data[s]['low'])) > 0.618).astype(int)
			bull_bearcpin = (((data[s]['close'] - data[s]['low'])/(
				data[s]['high'] - data[s]['low'])) > 0.618).astype(int)
			data[s]['Up pin_bar'] = ((bull_bullcpin & bull_bearcpin & data[s]['diff'] > data[s]['std'])).astype(int)

			bear_bullcpin = ((data[s]['high'] - data[s]['open'])/(
				data[s]['high'] - data[s]['low'])) > 0.618
			bear_bearcpin = ((data[s]['high'] - data[s]['close'])/(
				data[s]['high'] - data[s]['low'])) > 0.618
			data[s]['Down pin_bar'] = (bear_bullcpin & bear_bearcpin & (data[s]['diff'] > data[s]['std'])).astype(int)

			data[s].drop(['diff', 'std'], axis = 1, inplace = True)

			for i in range(self.start_bars, self.prev_bars, self.period):
				data[s]['Down pin_bar {}'.format(i + self.period)] = data[s]['Down pin_bar'].shift(i + self.period)
				data[s]['Up pin_bar {}'.format(i + self.period)] = data[s]['Up pin_bar'].shift(i + self.period)

		print("Pin Bar took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def bar_perc(self, data):

		print("Getting bar percentages...")

		start = time.perf_counter()
		for s in self.symbols:
			data[s]['body'] = ((data[s]['close'] - data[s]['open']) / (data[s]['high'] - data[s]['low'])).abs()
			data[s]['Up'] = ((data[s]['close'] - data[s]['open']) > 0).astype(int)
			for i in range(0, self.prev_bars, self.period):
				data[s]['body {}'.format(i + self.period)] = data[s]['body'].shift(i + self.period)
				data[s]['Up {}'.format(i + self.period)] = data[s]['Up'].shift(i + self.period)


		print("Bar percentages took: {:0.4f}\n".format(time.perf_counter() - start))
		return data

	def price_speed(self, data):

		print("Getting speed of bars...")
		start = time.perf_counter()

		for s in self.symbols:
			for i in range(self.start_bars, self.prev_bars, self.period):
				data[s]['Speed / {}bars'.format(i + self.period)] = data[s]['close'] - data[s]['close'].shift(i + self.period)

		print("Speed took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def bar_differences(self, data):

		print("Taking bar ohlc differences...")
		start = time.perf_counter()

		for s in self.symbols:
			data[s]['high - low'] = data[s]['high'] - data[s]['low']
			data[s]['close - open'] = data[s]['close'] - data[s]['open']
			data[s]['high - open'] = data[s]['high'] - data[s]['open']
			data[s]['low - open'] = data[s]['low'] - data[s]['open']
			data[s]['high - close'] = data[s]['high'] - data[s]['close']
			data[s]['low - close'] = data[s]['low'] - data[s]['close']

			for i in range(self.start_bars, self.prev_bars, self.period):
				data[s]['high - low {}'.format(i + self.period)] = data[s]['high - low'].shift(i + self.period)
				data[s]['close - open {}'.format(i + self.period)] = data[s]['close - open'].shift(i + self.period)
				data[s]['high - open {}'.format(i + self.period)] = data[s]['high - open'].shift(i + self.period)
				data[s]['low - open {}'.format(i + self.period)] = data[s]['low - open'].shift(i + self.period)
				data[s]['high - close {}'.format(i + self.period)] = data[s]['high - close'].shift(i + self.period)
				data[s]['low - close {}'.format(i + self.period)] = data[s]['low - close'].shift(i + self.period)

		print("Bar Diff took: {:0.4f}\n".format(time.perf_counter() - start))

		return data

	def create_target(self, data):
		# It is likely better to combine the basic features first rather than after getting new features.

		print("Creating Target...\n")

		if self.kind == 'Regression':
			for s in self.symbols:
				data[s]['Target'] = data[s]['close'].shift(-1)

		if self.kind == 'Classification':
			sym_dict = {}

			for s in self.symbols:
				self.start_time = time.perf_counter()
				print("Loading {}...".format(str(s).upper()))
				sym_dict[s] = []

				for i in range(0, len(data[s])):
					if data[s]['pct fut'][i] > 0.5:
						sym_dict[s].extend([1])
					elif data[s]['pct fut'][i] < -0.5:
						sym_dict[s].extend([-1])
					else:
						sym_dict[s].extend([0])
				print("* {0} Completed in {1:0.4f} seconds\n".format(str(s).upper(), time.perf_counter() - self.start_time))

			self.start_time = time.perf_counter()
			print("Concatinating Targets...".format())
			for s in self.symbols:        
				data[s] = pd.concat([data[s], pd.DataFrame(sym_dict[s], index = data[s].index, columns = ['Target'])], axis = 1)

			print("Target concatination completed in {:0.4f} seconds\n".format(time.perf_counter() - self.start_time))

		return data

	def clean_data(self, data, combine):

		target_list = []
		for s in self.symbols:
		    try:
		        data[s].drop(['pct fut'], axis = 1, inplace = True)
		    except KeyError:
		        pass
		            
		for s in self.symbols:
		 	data[s].dropna(inplace = True)

		if combine:
			all_data = []
			for k in data.keys():
				all_data.append(data[k])
			data = pd.concat(all_data).dropna()

		return data
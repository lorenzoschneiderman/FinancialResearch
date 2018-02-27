
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
from scipy import stats
import Tapack
import Index ## not in use
Benchmark = Index.return_daily_index() ## not in use

class KNNClassifier:

	def __init__(self, mode):
		self.mode = mode
		self.partition_ratio = map(float, (4,1,1)) ## train:validate:test ratio
		self.window_size = 21
		self.Train_X = []
		self.Train_Y = [] # [ [theoretical, market-adjusted]]
		self.Validate_X = [] # validation merges with Train_X after validation 
		self.Validate_Y = [] # merges with training set after validating too
		self.Test_X = []
		self.Test_Y = []
		self.Test_dates = []
		self.Test_stocks = []

		pass

	## given a dict of timeseries model memorizes data
	def format_examples(self, data):

		for stock in data.keys():
			price_timeseries = data[stock]['close'] # pd.Series
			price_series = data[stock]['close'].values # np.Array
			date_series = data[stock]['close'].index.values # ary
			objective_series = [-1] + (np.diff(price_series)/price_series[:-1]).tolist() # ary of price changes daily
			assert len(objective_series) == len(price_series) and len(price_series) == len(date_series)
			
			if self.mode == 'MACD':
				macd_series = TApack.returnMACD(price_series) # ary # returns same len array filled with nans in beginning
				X, Y, dates = convolute(macd_series, np.array(objective_series), self.window_size, date_series)
			if self.mode == 'RSI':
				rsi_series = TApack.returnRSI(price_series)
				X, Y, dates = convolute(rsi_series, np.array(objective_series), self.window_size, date_series)
			if self.mode == 'PRICE':
				X, Y, dates = convolute(price_series, np.array(objective_series), self.window_size, date_series, normalize_price=True)
			
			inc = 0 ## remove examples with 'nan' completely
			for day in range(len(X)):
				if np.isnan(X[day]).any():
					inc +=1
				else:
					break
			X, Y, dates = X[inc:], Y[inc:], dates[inc:]

			train_prop = self.partition_ratio[0] / (self.partition_ratio[0] + self.partition_ratio[1] + self.partition_ratio[2])
			validate_prop = self.partition_ratio[1] / (self.partition_ratio[0] + self.partition_ratio[1] + self.partition_ratio[2])
			test_prop = self.partition_ratio[2] / (self.partition_ratio[0] + self.partition_ratio[1] + self.partition_ratio[2])
			
			self.Train_X += X[:int(train_prop*len(X))]
			self.Train_Y += Y[:int(train_prop*len(X))]
			self.Validate_X += X[int(train_prop*len(X)):int(train_prop*len(X))+int(validate_prop*len(X))]
			self.Validate_Y += Y[int(train_prop*len(X)):int(train_prop*len(X))+int(validate_prop*len(X))]
			self.Test_X += X[int(train_prop*len(X))+int(validate_prop*len(X)):]
			self.Test_Y += Y[int(train_prop*len(X))+int(validate_prop*len(X)):]

			self.Test_dates += dates[int(train_prop*len(X))+int(validate_prop*len(X)):] # date and stock identifier of each prediction in test set
			self.Test_stocks += data[stock]['ticker'] * len(self.Test_dates)

	## optimize hyperparameter K
	def validate(self):

		Potential_K = range(1,400,5)
		Potential_K = range(255, 260, 1)
		R_Sq_Score = []
		Theoretical_History = []  # list(empirical outcomes)
		Prediction_History = {} # {k: list(predictions)}
		for k in Potential_K:
			Prediction_History[k] = []

		for example in range(len(self.Validate_X)): # make predictions for every example in validation set
			features = self.Validate_X[example] 
			Theoretical_History.append(self.Validate_Y[example]) # emperical outcome
			predictions = self.predict([features], Potential_K)[0] # [pred-k1, pred-k2, pred-k3,... ]
			for numk in range(len(Potential_K)):
				Prediction_History[Potential_K[numk]].append(predictions[numk][0])
		for k in Potential_K:
			r_value = np.corrcoef(Prediction_History[k], Theoretical_History)[0][1]
			R_Sq_Score.append(r_value)

		print "R_Sq_Score"
		print R_Sq_Score
		print "Potential_K"
		print Potential_K

		plt.figure(0)
		plt.plot(R_Sq_Score)
		plt.xlabel("Potential_K")

		highest_index = R_Sq_Score.index(max(R_Sq_Score))
		self.Best_K = Potential_K[highest_index]

		self.Train_X += self.Validate_X ## adds validation examples to "training" for test-set evaluation 
		self.Train_Y += self.Validate_Y

		return self.Best_K

	# evaluate correlation coefficient after k optimized
	def test(self):
		Prediction_History = []
		Theoretical_History = []
		Ordered_Stocks = []
		for example in range(len(self.Test_X)):
			features = self.Test_X[example]
			prediction = self.predict([features], [self.Best_K])[0][0][0]
			Prediction_History.append(prediction)
			Theoretical_History.append(self.Test_Y[example])
			Ordered_Stocks.append(self.Test_stocks[example])

		r_value = np.corrcoef(Prediction_History, Theoretical_History)[0][1]
		print "Prediction_History"
		print Prediction_History
		print "Theoretical_History"
		print Theoretical_History
		print "final test r_value: ", r_value, " of sample size: ", len(self.Validate_Y)
		
		plt.figure(1)
		plt.scatter(Prediction_History, Theoretical_History)
		plt.show()
		# assert len(Prediction_History) == len(self.Test_dates)

		return Prediction_History, Theoretical_History, self.Test_dates, Ordered_Stocks # self.Test_dates and Ordered_Stocks used to identify transactions


	# X - list of query-examples, K - list of trial-ks (optimize runtime)
	def predict(self, X, K):
		hypotheses = [] # ordered predictions for examples in X
		for example in X:
			DiffsSquared = np.sum(np.square(np.array(example) - self.Train_X), axis=1)	
			Distance_Outcome_Pairs = np.append(DiffsSquared.reshape((-1, 1)), np.array(self.Train_Y).reshape((-1, 1)), axis=1).tolist() ## [[distance, outcome], ...] for every example in training
			Sorted_Similar_Examples = sorted(Distance_Outcome_Pairs) # sort by distance
			k_outputs = []
			for k in K: # find averages for each 
				Most_Similar_Examples = Sorted_Similar_Examples[:k] # only nearest k neighbors
				Nearest_Outcomes = [x[1] for x in Most_Similar_Examples]
				point_estimate = np.mean(Nearest_Outcomes) 
				uncertainty = Nearest_Outcomes ## uncertainty is std of all nearest outcomes (or simply a list storing them all)
				discrepancy = np.mean([x[0] for x in Most_Similar_Examples]) ## discrepancies is avg of distance
				k_outputs.append([point_estimate, -1, -1]) # uncertainty, and discrepancy are not evaluated in this test
			hypotheses.append(k_outputs)
		return hypotheses


## returns pairs of X, Y given continuous pd.Series for the features and objectives. In other words, stagnates each objective to correspond to the previous day's observation of feature
def convolute(feature_series, objective_series, window_size, date_series, normalize_price = False):
	X = []
	Y = []
	normalize_market = False ## when on: subtracts beta(constant) * S&P-500's-change from every examples empirical-outcome
	exclude_days = 0 # number of days to be excluded because of magnitude of price change
	include_days = 0
	dates = []
	for d in range(window_size - 1, feature_series.size - 1):
		
		x = feature_series[d+1-window_size:d+1].tolist()
		y = objective_series[d + 1]
		date = date_series[d+1]

		if normalize_price: # makes x expresses arry of prices as a percentage of current (last) price 
			x = np.array(x)
			x = (x - x[-1])/x[-1] # normalize to most recent price
			x = x.tolist()

		if normalize_market: # makes y express excess gains to index
			try:
				beta = .5
				market_change = Benchmark[pd.to_datetime(date).strftime('%Y-%m-%d')]
				y = y - (market_change * beta)
			except Exception, e:
				print e
				print "failed to find snp change on day ", date
				pass
			
		if abs(y) < .037: ## doesn't add examples to dataset if they are the day preceeding highly volatile movement 
			X.append(x)
			Y.append(y)
			dates.append(date)
			include_days += 1
		else:
			exclude_days += 1
	print exclude_days, " out of ", include_days + exclude_days, " excluded for volaitility"
	return X, Y, dates

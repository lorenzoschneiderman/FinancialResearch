
from Yahoofetcher import get_historical_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from dateutil.parser import parse
import time


def make_data_set(N_samples, ticker_universe, name, delete_data = False):

	stock_data = {} # returned ticker: {'close': priceseries, 'misc:' idk} 
	dateindex = [] # ordered dates of stock-price-sample observation 
	history = {}
	for ticker in ticker_universe:
		time.sleep(0.6)
		try:
			sample = get_historical_data(ticker, N_samples) ## returns list of dictionaries 
			history[ticker] = sample
		except Exception, e:
			print e
			print "failed to retrieve data, removed ticker from universe: ", ticker
			ticker_universe.remove(ticker)

	for ticker in ticker_universe:
		price_series = []
		date_series = []
		try:
			for day in history[ticker]:
				price_series.append(day['close'])
				date_series.append(datetime.strptime(day['date'], '%Y-%m-%d'))
			price_timeseries = pd.Series(price_series, index=date_series)
			stock_data[ticker] = {'close':[],'misc':[]} ## close to be a timeseries and misc 2-d array
			stock_data[ticker]['close'] = price_timeseries
			stock_data[ticker]['ticker'] = ticker
		except Exception, e:
			print e
			print "failed to make dataset for stock: ", ticker ## should not happen
	
	if delete_data:
		"Deleting data and returning new dataset named ", name
		NewDict = {name: stock_data}
		information = ["N_samples", N_samples, "len(ticker_universe)", len(ticker_universe)] # stores specs of data download
		NewDict[name+'.info'] = information
		pickle.dump(NewDict , open( "DataStorage.p", "wb" )) ## stores database in DataStorage.pickle
		return NewDict
	else:
		"Keeping old data and adding new dataset named ", name
		try:
			PrevData = pickle.load(open("DataStorage.p", 'r'))
		except Exception, e:
			print e
			print "Couldn't open old data, returning dataset named", name
			PrevData = {}
		PrevData[name] = stock_data
		information = ["N_samples", N_samples, "len(ticker_universe)", len(ticker_universe)]
		PrevData[name+'.info'] = information
		pickle.dump(PrevData , open( "DataStorage.p", "wb" )) ## stores database in DataStorage.pickle
		return PrevData

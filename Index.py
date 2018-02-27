
import numpy as np
import csv


reader = csv.DictReader(open('GSPC.csv', 'rb'))  # SPY data from 2017/01/01 to 2018/02/02
dataseries = list((line for line in reader))
daily_gains = {} # {str(date): mkt's-percent-change}
snp = []
for i in range(1, len(dataseries)):
	daily_gains[dataseries[i]['Date']] = (float(dataseries[i]['Close']) - float(dataseries[i-1]['Close'])) / float(dataseries[i-1]['Close'])
	snp.append(daily_gains[dataseries[i]['Date']])

def return_daily_index():
	return daily_gains

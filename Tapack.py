
import talib
import numpy as np


def returnMACD(price_ary):
	return talib.MACD(price_ary)[0]/price_ary ## changed tolist

def returnRSI(price_ary):
	return talib.RSI(price_ary)/price_ary[-1]

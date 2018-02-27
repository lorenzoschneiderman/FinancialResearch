
from Datafeed import make_data_set
from StocksKNN import KNNClassifier 
import numpy as np
import pickle
import matplotlib.pyplot as plt

making_data_set = False ## if true, downloads all 500 stocks on ticker list onto Datastorage.p
if making_data_set:
	num_days = 200
	ticker_list = ['EVHC', 'MAA', 'COTY', 'COO', 'CHTR', 'MTD', 'ALB', 'LNT', 'FBHS', 'TDG', 'AJG', 'LKQ', 'DLR', 'ALK', 'AYI', 'GPN', 'ULTA', 'FL', 'CNC', 'HOLX', 'UAA', 'UDR', 'AWK', 'CXO', 'FRT', 'CFG', 'EXR', 'WLTW', 'CHD', 'CSRA', 'ILMN', 'SYF', 'HPE', 'VRSK', 'CMCSA', 'NWS', 'FOX', 'UAL', 'ATVI', 'SIG', 'PYPL', 'AAP', 'KHC', 'JBHT', 'QRVO', 'O', 'AAL', 'EQIX', 'HBI', 'SLG', 'HSIC', 'SWKS', 'HCA', 'RCL', 'URI', 'UHS', 'DISCK', 'MLM', 'AMG', 'XEC', 'AVGO', 'NAVI', 'UA', 'GOOGL', 'ESS', 'TSCO', 'ADS', 'FB', 'MHK', 'GGP', 'ALLE', 'KORS', 'AME', 'VRTX', 'DAL', 'NWSA', 'NLSN', 'FOXA', 'ZTS', 'GM', 'KSU', 'MAC', 'REGN', 'PVH', 'ABBV', 'DLPH', 'GRMN', 'DG', 'MDLZ', 'PNR', 'LYB', 'STX', 'LRCX', 'MNST', 'ALXN', 'KMI', 'PSX', 'CCI', 'TRIP', 'BWA', 'DLTR', 'PRGO', 'XYL', 'TEL', 'MOS', 'ACN', 'MPC', 'CMG', 'BLK', 'EW', 'FFIV', 'NFLX', 'NFX', 'IR', 'JCI', 'CB', 'KMX', 'CERN', 'OKE', 'DISCA', 'HP', 'BRK.B', 'NRG', 'ROP', 'ROST', 'V', 'PCLN', 'FAST', 'FMC', 'RHT', 'PWR', 'WDC', 'FTI', 'ORLY', 'HRL', 'VTR', 'HCN', 'IRM', 'FLIR', 'SCG', 'EQT', 'RSG', 'SRCL', 'XRAY', 'WYNN', 'PBCT', 'SJM', 'WEC', 'NDAQ', 'DPS', 'FLS', 'APH', 'PXD', 'HRS', 'CRM', 'CF', 'IVZ', 'DVA', 'MA', 'SNI', 'COG', 'ISRG', 'HCP', 'PM', 'TSS', 'RRC', 'AMT', 'JEC', 'EXPD', 'NBL', 'EXPE', 'ANDV', 'ICE', 'MCHP', 'LUK', 'AKAM', 'DFS', 'AIZ', 'HST', 'CHRW', 'VAR', 'RL', 'AVB', 'CTSH', 'CBG', 'FIS', 'CELG', 'KIM', 'BXP', 'VRSN', 'EL', 'AMZN', 'PDCO', 'LEN', 'AMP', 'PSA', 'STZ', 'NOV', 'LH', 'GILD', 'MYL', 'ETFC', 'ESRX', 'PLD', 'SYMC', 'DGX', 'TRV', 'MON', 'EA', 'GS', 'PFG', 'PRU', 'UPS', 'SPG', 'EQR', 'NVDA', 'ABC', 'ZBH', 'COL', 'FISV', 'CTAS', 'SYK', 'INTU', 'RHI', 'EOG', 'DVN', 'TIF', 'A', 'XLNX', 'VMC', 'BBY', 'NTAP', 'RF', 'YUM', 'PGR', 'EFX', 'ADBE', 'BSX', 'MU', 'CBS', 'LUV', 'UNH', 'MSFT', 'KEY', 'UNM', 'EMN', 'CSCO', 'COST', 'IPG', 'PX', 'AMGN', 'AEE', 'MRO', 'ADSK', 'ORCL', 'NWL', 'TMK', 'ECL', 'NKE', 'C', 'STI', 'PNC', 'HD', 'AVY', 'MMC', 'CA', 'SYY', 'HRB', 'MDT', 'GPS', 'JWN', 'ITW', 'PH', 'DOV', 'TJX', 'CNP', 'NOC', 'PKI', 'APD', 'NUE', 'BLL', 'HAS', 'LMT', 'HES', 'PHM', 'LOW', 'T', 'VZ', 'LB', 'CAG', 'OXY', 'AAPL', 'BF.B', 'SNA', 'SWK', 'WMT', 'MAT', 'ADM', 'GWW', 'MAS', 'ADP', 'FDX', 'PCAR', 'AIG', 'FLR', 'WBA', 'VFC', 'TXT', 'INTC', 'TGT', 'AET', 'AXP', 'BAC', 'CI', 'DUK', 'LNC', 'TAP', 'NEE', 'DIS', 'WFC', 'IFF', 'BCR', 'JPM', 'WMB', 'HPQ', 'GPC', 'JNJ', 'BAX', 'BDX', 'LLY', 'MCD', 'NEM', 'CLX', 'GIS', 'CSX', 'CMI', 'EMR', 'SLB', 'SHW', 'ABT', 'ARNC', 'HON', 'MMM', 'AES', 'AFL', 'AGN', 'ALL', 'GOOG', 'MO', 'AEP', 'APC', 'ADI', 'ANTM', 'AON', 'APA', 'AIV', 'AMAT', 'AZO', 'BHGE', 'BK', 'BBT', 'BIIB', 'BA', 'BMY', 'CPB', 'COF', 'CAH', 'CCL', 'CAT', 'CTL', 'SCHW', 'CHK', 'CVX', 'CINF', 'CTXS', 'CME', 'CMS', 'COH', 'KO', 'CL', 'CMA', 'COP', 'ED', 'GLW', 'CVS', 'DHI', 'DHR', 'DRI', 'DE', 'D', 'DWDP', 'DTE', 'ETN', 'EBAY', 'EIX', 'ETR', 'ES', 'EXC', 'XOM', 'FITB', 'FE', 'F', 'BEN', 'FCX', 'GD', 'GE', 'GT', 'HAL', 'HOG', 'HIG', 'HSY', 'HUM', 'HBAN', 'IBM', 'IP', 'JNPR', 'K', 'KMB', 'KLAC', 'KSS', 'KR', 'LLL', 'LEG', 'L', 'MTB', 'M', 'MAR', 'MKC', 'MCK', 'MRK', 'MET', 'MCO', 'MS', 'MSI', 'NI', 'NSC', 'NTRS', 'OMC', 'PAYX', 'PEP', 'PFE', 'PCG', 'PNW', 'PPG', 'PPL', 'PG', 'PEG', 'QCOM', 'RTN', 'ROK', 'SEE', 'SRE', 'SO', 'SPGI', 'SBUX', 'STT', 'TROW', 'TXN', 'TMO', 'TWX', 'TSN', 'USB', 'UNP', 'UTX', 'VLO', 'VIAB', 'VNO', 'WM', 'WAT', 'WU', 'WRK', 'WY', 'WHR', 'WYN', 'XEL', 'XRX', 'XL', 'ZION']
	name = "test-short"
	data = make_data_set(num_days, ticker_list, name, delete_data = False)
else:
	data = pickle.load(open("DataStorage.p", 'r'))

data_source = data['test-short']

# MacdKNN = KNNClassifier('MACD')
# RsiKNN = KNNClassifier('RSI')
PriceKNN = KNNClassifier('PRICE')
PriceKNN.format_examples(data_source)
PriceKNN.validate()
PriceKNN.test()

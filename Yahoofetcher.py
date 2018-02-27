
from BeautifulSoup import BeautifulSoup as bs
import urllib
import urllib2
import csv
#   returns list of dictionaries with daily date, close, volume sampling
#   list in order of increasing date
def get_historical_data(name, numDays):

    the_url = 'https://query1.finance.yahoo.com/v7/finance/download/' + name + '?'
    #          https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=&period2=&interval=1d&events=history&crumb=iaEbJM1y1zC
    values = {'period1':'1360040400','period2':'1517806800', 'interval':'1d', 'events':'history', 'crumb':'iaEbJM1y1zC'}
    data = urllib.urlencode(values)
    req = urllib2.Request(the_url, data)
    handle = urllib2.urlopen(req)
    the_page = handle.read()
    data = the_page.split("\n")[1:-1]
    # print "header: ", the_page.split("\n")[0]
    
    history = [] ## list of data points
    for each_row in data[-numDays:]: # first/top row is most recent date
        days_terms = each_row.split(",")
        sample = {} # days close and volume date dict
        if type(days_terms) == type([]) and len(days_terms) == 7:
            sample['date'] = days_terms[0]
            sample['close'] = float(days_terms[4])
            sample['volume'] = int(days_terms[6])
            history.append(sample)
        else:
            print 'skipped day'
            pass

    print name, " stock has ", len(data), " days"
    return history


# print get_historical_data('AMZN', 3)

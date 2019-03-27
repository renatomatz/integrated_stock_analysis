import quandl
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import pdb

with open("config.txt") as config:
    company = config.readline()[8:-1]
    ticker = config.readline()[7:-1]
    pdb.set_trace()
    end_date = datetime.strptime(config.readline()[12:-1], "%Y-%m-%d")
    start_date = end_date - timedelta(days=700)
    end_date = str(end_date)[9:]
    start_date = str(start_date)[9:]


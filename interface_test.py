from quandl_interface import *

xom = Equity("XOM", market="SPY", start=None, end=None, key="key.txt")

# xom.enable()

# xom.set_start("2013")

# xom.set_end("2018")

# xom.set_market("SPY")

# get_dataset = xom.get_dataset("SF1", columns=["calendardate", "ebitda"], save_new=True)

# xom.update_dataset("SF1")

# xom.available_saved()

# xom._exists("SF1")

# xom._make_path("SF1")

# _ts_index = xom._ts_index(get_dataset)

# _filter_dates = xom._filter_dates(_ts_index)

# _round_col = xom._round_col(_filter_dates, ["ebitda"])

get_daily_metrics = xom.get_daily_metrics()

# get_fundamentals = xom.get_fundamentals()

# get_events = xom.get_events()

# get_industry_prices = xom.get_industry_prices()

# # get_industry = xom.get_industry()

# xom.set_industry("BPT")

# get_comps = xom.get_comps()


xomC = Comps(["XOM", "VZ", "UTX"], start=None, end=None, key="key.txt", market="SPY")

# xomC.enable()

# xomC.set_start("2013")

# xomC.set_end("2018")

# xomC.set_market("SPY")

# get_dataset = xomC.get_dataset("SF1", columns=["calendardate", "ticker", "ebitda"], save_new=True)

# xomC.update_dataset("SF1")

# xomC.available_saved()

# xomC._exists("SF1")

# xomC._make_path("SF1")

# _ts_index = xomC._ts_index(get_dataset)

# _filter_dates = xomC._filter_dates(_ts_index)

# _round_col = xomC._round_col(_ts_index, ["ebitda"])

# get_daily_metrics = xomC.get_daily_metrics()

# get_fundamentals = xomC.get_fundamentals()

# get_events = xomC.get_events()

# get_industry = xomC.get_industry()

# get_industry_prices = xomC.get_industry_prices()

# get_comps = xomC.get_comps()

 
cust = Custom(mem_file="SPY", key="key.txt")

# get_yahoo_market_index = cust.get_yahoo_market_index("SPY", columns=["Date", "Close"])


xomG = EquityGrapher(line_width=1.3, alpha=0.6, colors=None)

get_price = xomG.get_price(xom)

fundamentals_chart = xomG.fundamentals_chart(xom, save=True)
 
fundamentals_graph = xomG.fundamentals_graph(xom, save=True)

sales_growth = xomG.sales_growth(xom, save=True)

events = xomG.events(xom, save=True)


xomCG = CompsGrapher(line_width=1.3, alpha=0.7, colors=None)

fundamentals_chart = xomCG.fundamentals_chart(xomC, save=True)

comps_graph = xomCG.comps_graph(xomC, mkt=cust, save=True)

print("Done")
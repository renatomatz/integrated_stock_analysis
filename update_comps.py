import import_data
import pandas as pd

config = import_data.get_config("test_files/config.txt")
comps = import_data.get_comps(config)
comp_funds = import_data.get_comp_fundamentals([*comps["ticker"]])

comp_funds_latest = comp_funds.groupby("ticker").apply(lambda x: x[x["calendardate"] == max(x["calendardate"])])
# leave only latest metrics
comp_funds_latest.index = comp_funds_latest.index.droplevel()

comps_merged = pd.merge(left=comp_funds_latest, right=comps, 
                        on=["ticker"],
                        how="left")

comps_merged = comps_merged[["name", "ticker", "price", "marketcap", "ev/ebitda", "ev/sales", "ev/fcf"]]
# maintail relevant columns
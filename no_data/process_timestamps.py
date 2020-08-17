import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("no_pd_cfs.csv")
    ts_fields = ['TimeCreate','TimeDispatch','TimeArrival','TimeClosed']
    for ts in ts_fields:
        df[ts] = pd.to_datetime(df[ts], infer_datetime_format=True)
        df["{}_epoch".format(ts)] = df[ts].apply(lambda x: x.timestamp() if not pd.isnull(x)  else x)
    df.to_csv("formatted_no_cfs.csv")

import pandas as pd
from scipy.stats import poisson
df = pd.read_csv("formatted_no_cfs.csv")
times = pd.to_datetime(df['TimeCreate'])
a = df.groupby([times.dt.year, times.dt.month, times.dt.week, times.dt.day, times.dt.hour])['NOPD_Item'].count() # Number of entries for every hour
N = a.shape[0]
buckets = {}
for x in range(a.min(), a.max()+1):
    buckets[x] = a[a==x].count()

mean = sum([(k*v) for k,v in buckets.items()])/N
print(buckets) 

p = poisson(mean)
expected = {k: (p.pmf(k)*N) for k in buckets.keys()}
print(expected)

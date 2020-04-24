from math import sqrt
import csv
import pandas as pd
import dask.dataframe as dd
import scipy.stats as st
import numpy as np
import json
import itertools
import os.path
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import scipy.signal as ss
from statsmodels.stats.diagnostic import lilliefors
from scipy import optimize
from tqdm import tqdm_notebook

class Probability_Statistics():
    def __init__(self, data_name, data_path, data_map_path="/home/blakemoss/911_modeling/ts_data_map.csv"):
        plt.rcParams.update({'figure.max_open_warning': 0})
        self.event_names = ["call_received", "entered_to_cad", "first_unit_assigned", "first_unit_enroute", "first_unit_onscene",
                             "first_unit_tohospital", "first_unit_athospital", "incident_closed"]
    
        self.intervals = [("call_received", "entered_to_cad"), ("entered_to_cad", "first_unit_assigned"),
                            ("call_received", "first_unit_assigned"),
                            ("first_unit_assigned", "first_unit_enroute"), ("first_unit_enroute", "first_unit_onscene"),
                            ("first_unit_tohospital", "first_unit_athospital"),
                            ("first_unit_athospital", "incident_closed"),("first_unit_onscene", "incident_closed")]
 
        self.name = data_name
        self.data_path = data_path
        self.map_path = data_map_path
        self.source_df = pd.read_csv(self.map_path)    
        self.source_df = self.source_df[self.source_df['Dataset']==data_name]
        self.source_map = json.loads(self.source_df.to_json(orient='records'))[0]
        self.unique_id = self.source_map['unique_id']
        self.call_type_field = self.source_map['call_type_field']
        self.priority_type_field = self.source_map['priority_type_field']
        self.hour_ranges = []
        for x in range(0,23):
            range_ = ("{}:00".format(x), "{}:00".format(x+1))
            self.hour_ranges.append(range_)
        self.hour_ranges.append(("23:00", "0:00"))
        if self.source_map['Dtype'] is not None:
            self.dtype = json.loads(self.source_map['Dtype'])
        else:
            self.dtype = None
        self.num_days = None
 

    def exponential_fit_per_hour_interarrival(self):
        hour_stats = {event_name:"Not Recorded" for event_name in self.event_names}
        min_stats = {event_name:"Not Recorded" for event_name in self.event_names}
        for event_name in tqdm_notebook(self.event_names):
            mapped_name = self.source_map[event_name]
            if mapped_name is not None:
                all_columns = [self.unique_id, mapped_name]
                df = dd.read_csv(self.data_path, usecols=all_columns)
                df = df.dropna()
                df = df.compute()
                df[mapped_name] = pd.to_datetime(df[mapped_name], errors="coerce", infer_datetime_format=True)
                df = df.sort_values(by=mapped_name,ascending=True)
                df.index = df[mapped_name]
                df['inter_arrival'] = (df[mapped_name]-df[mapped_name].shift()).dt.seconds.fillna(np.float64(0))
                zero_inter_arrival = df[df['inter_arrival']==0].index
                df.drop(zero_inter_arrival, inplace=True)
                reject = 0
                not_reject = 0
                print("--------------------------------------------------------------------")
                dists = {"Exponential": 0, "Lomax":0, "Pareto": 0, "ExponWeibull":0, "Beta":0}
                for x in self.hour_ranges:
                    #fig, ax = plt.subplots()
                    hour_slice = df.between_time(*x)  
                    total_counts = hour_slice['inter_arrival'].values
                    #histo, bin_edges, patches = ax.hist(total_counts, bins=100, density=False)
                    #args = st.expon.fit(total_counts)
                    #cdf = st.expon.cdf(bin_edges, *args)
                    #expected_values = len(total_counts) * np.diff(cdf)
                    #ax.plot(bin_edges[:-1], expected_values)
                    ks, pval = lilliefors(total_counts, dist='exp', pvalmethod='table') 
                    if pval < .05:
                        #print("{}:{} -> Not from exponential".format(event_name, x))                    
                        reject += 1
                    else:
                        #print("{}:{} -> From exponential".format(event_name, x))                    
                        not_reject += 1
                    res = self.test_chi_squared(total_counts)
                    if res is not None:
                        dists[list(res.keys())[0]] += 1
                print("{}: {} Not reject, {} reject".format(event_name, not_reject, reject))
                print(sorted(dists.items(), key=lambda x: x[1]))
                print("--------------------------------------------------------------------")

    def exponential_fit_per_hour_interval(self):
#        hour_stats = {event_name:"Not Recorded" for event_name in self.event_names}
#        min_stats = {event_name:"Not Recorded" for event_name in self.event_names}
        interval_stats = {"{}-{}".format(start, end):"Not Recorded" for start, end in self.intervals}
        for start_event, end_event in tqdm_notebook(self.intervals):
            start_mapped_name = self.source_map[start_event]
            end_mapped_name = self.source_map[end_event]
            if start_mapped_name is not None and end_mapped_name is not None:
                all_columns = [self.unique_id, start_mapped_name, end_mapped_name]
                df = dd.read_csv(self.data_path, usecols=all_columns)
                df = df.dropna()
                df = df.compute()
                df[start_mapped_name] = pd.to_datetime(df[start_mapped_name], errors="coerce", infer_datetime_format=True)
                df[end_mapped_name] = pd.to_datetime(df[end_mapped_name], errors="coerce", infer_datetime_format=True)
                df.index = df[start_mapped_name]
                df['delta'] = (df[end_mapped_name]-df[start_mapped_name]).dt.seconds.fillna(np.float64(0))
                zero_interval = df[df['delta']==0].index
                df.drop(zero_interval, inplace=True)
                total_counts = df['delta'].values
                ks, pval = lilliefors(total_counts, dist='exp', pvalmethod='table') 
                reject = 0
                not_reject = 0
                if pval < .05:
                    print("{}-{}:{} -> Not from exponential".format(start_event, end_event, "Overall"))                    
                    reject += 1
                else:
                    print("{}-{}:{} -> From exponential".format(start_event, end_event, "Overall"))                    
                    not_reject += 1
                dists = {"Exponential": 0, "Lomax":0, "Pareto": 0, "ExponWeibull":0}
                for x in self.hour_ranges:
                    fig, ax = plt.subplots()
                    ax.set_title("{} Histogram for Time Elapsed between {} and {} events Between {}".format(self.name, start_event, end_event, x)) 
                    ax.set_xlabel("Seconds")
                    ax.set_ylabel("Frequency")
                    hour_slice = df.between_time(*x)  
                    total_counts = hour_slice['delta'].values
                    histo, bin_edges, patches = ax.hist(total_counts, bins=100, density=False)
                    args = st.expon.fit(total_counts)
                    cdf = st.expon.cdf(bin_edges, *args)
                    expected_values = len(total_counts) * np.diff(cdf)
                    ax.plot(bin_edges[:-1], expected_values, label="Exponential Fit")
                    ax.legend()
                    ks, pval = lilliefors(total_counts, dist='exp', pvalmethod='table') 
                    if pval < .05:
                        #print("{}-{}:{} -> Not from exponential".format(start_event, end_event, "Overall"))                    
                        reject += 1
                    else:
                        print("{}-{}:{} -> From exponential".format(start_event, end_event, "Overall"))                    
                        not_reject += 1
                    #res = self.test_chi_squared(total_counts)
                    #if res is not None:
                    #    dists[list(res.keys())[0]] += 1
                print("{}-{}: {} Not reject, {} reject".format(start_event, end_event, not_reject, reject))
                #print(sorted(dists.items(), key=lambda x: x[1]))
                print("--------------------------------------------------------------------")

    def test_chi_squared(self, count):
        histo, bin_edges = np.histogram(count, bins=100, density=False)
        dists = {"Exponential": st.expon, "Lomax":st.lomax, "Pareto": st.pareto, "Beata": st.beta, "ExponWeibull":st.exponweib}
        best_pval = 0
        best_dist = None 
        for dist_name, dist in dists.items(): 
            args = dist.fit(count)
            cdf = dist.cdf(bin_edges, *args)
            expected_values = len(count) * np.diff(cdf)
            chi_stat, pval = st.chisquare(histo, f_exp=expected_values, ddof=len(args))
            if pval > best_pval:
                best_pval = pval
                best_dist = {dist_name:(chi_stat,best_pval)}
        return best_dist 



    def write_statistics(self, stats_file="/home/blakemoss/911_modeling/stationary_stats.csv"):
        pass

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
from statsmodels.tsa.stattools import acf 
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy import optimize
from tqdm import tqdm_notebook

class Stationary_Statistics():
    def __init__(self, data_name, data_path, data_map_path="/home/blakemoss/911_modeling/ts_data_map.csv"):
        plt.rcParams.update({'figure.max_open_warning': 0})
        self.event_names = ["call_received", "entered_to_cad", "first_unit_assigned", "first_unit_enroute", "first_unit_onscene",
                             "first_unit_tohospital", "first_unit_athospital", "incident_closed"]
        self.intervals = [("call_received", "entered_to_cad"), ("entered_to_cad", "first_unit_assigned"),
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
        for x in range(0,24):
            range_ = ("{}:00".format(x), "{}:59".format(x))
            self.hour_ranges.append(range_)
        if self.source_map['Dtype'] is not None:
            self.dtype = json.loads(self.source_map['Dtype'])
        else:
            self.dtype = None
        self.num_days = None
 

    def calculate_arrival_rates_per_hour(self):
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
                df.index = df[mapped_name]
                total_num = len(df[self.unique_id].unique())
                hour_groups = df.groupby([df.index.hour])
                min_groups = df.groupby([df.index.minute])
                hour_per = []
                hour_counts = []
                min_per = []
                min_counts = []
                for name, group in hour_groups:
                    count = group[self.unique_id].count()
                    percent = group[self.unique_id].count()/total_num
                    count_obj = {name:count}
                    percent_obj = {name: percent}
                    hour_per.append(percent_obj)
                    hour_counts.append(count_obj)
                hour_stats[event_name] = {"hour_percents": hour_per, "hour_counts": hour_counts}

                for name, group in min_groups:
                    count = group[self.unique_id].count()
                    percent = group[self.unique_id].count()/total_num
                    count_obj = {name:count}
                    percent_obj = {name: percent}
                    min_per.append(percent_obj)
                    min_counts.append(count_obj)
                min_stats[event_name] = {"min_percents": min_per, "min_counts": min_counts}
        return {"hour_stats": hour_stats, "min_stats": min_stats}

    def auto_correlation_arrival_counts(self):
        hour_stats = {event_name:"Not Recorded" for event_name in self.event_names}
        for event_name in tqdm_notebook(self.event_names):
            mapped_name = self.source_map[event_name]
            if mapped_name is not None:
                all_columns = [self.unique_id, mapped_name]
                df = dd.read_csv(self.data_path, usecols=all_columns)
                df = df.dropna()
                df = df.compute()
                df[mapped_name] = pd.to_datetime(df[mapped_name], errors="coerce", infer_datetime_format=True)
                df.index = df[mapped_name]
                total_counts = df.groupby([df.index.year, df.index.month, df.index.day, df.index.hour, df.index.minute])[self.unique_id].count().values
                print(self.name, event_name)
                fig, ax = plt.subplots()
                plot_acf(total_counts, ax=ax,title="", lags=60)
                ax.set_ylabel("Autocorrelation Coefficient")
                ax.set_xlabel("Lag")
                plt.show()
                #for x in self.hour_ranges:
                #    hour_slice = df.between_time(*x)  
                #    total_counts = hour_slice.groupby([hour_slice.index.year, hour_slice.index.month, hour_slice.index.day, hour_slice.                                                       index.hour])[self.unique_id].count().values
                #    plot_acf(total_counts, lags=60, title="{} Hourly Autocorrelation Event Counts {}:{}".format(self.name, event_name, x))

    def auto_correlation_compare(self):
        all_min = []
        all_hour = []
        for event_name in tqdm_notebook(self.event_names):
            mapped_name = self.source_map[event_name]
            if mapped_name is not None:
                all_columns = [self.unique_id, mapped_name]
                df = dd.read_csv(self.data_path, usecols=all_columns)
                df = df.dropna()
                df = df.compute()
                df[mapped_name] = pd.to_datetime(df[mapped_name], errors="coerce", infer_datetime_format=True)
                df.index = df[mapped_name]
                total_min = df.groupby([df.index.year, df.index.month, df.index.day, df.index.hour, df.index.minute])[self.unique_id].count().values
                total_hour = df.groupby([df.index.year, df.index.month, df.index.day, df.index.hour])[self.unique_id].count().values
                print(self.name, event_name)
                acs_min = acf(total_min, fft=False, nlags=60)
                acs_hour = acf(total_hour, fft=False, nlags=24)
                mean_acs_min = np.mean(acs_min)
                mean_acs_hour = np.mean(acs_hour)
                all_min.append(mean_acs_min)
                all_hour.append(mean_acs_hour)

        return np.mean(all_min), np.min(all_hour)

    def auto_correlation_interarrival(self):
        hour_stats = {event_name:"Not Recorded" for event_name in self.event_names}
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
                zero_interval = df[df['inter_arrival']==0].index
                df.drop(zero_interval, inplace=True)
                total_counts = df['inter_arrival'].values
                plot_acf(total_counts, lags=10)
                t = acf(total_counts, lags=10)
                print(t)
                for x in self.hour_ranges:
                    hour_slice = df.between_time(*x)  
                    total_counts = hour_slice['inter_arrival'].values
                    plot_acf(total_counts, lags=10, title="{}:{}".format(event_name, x))


    def auto_correlation_intervals(self):
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
                df = df.sort_values(by=start_mapped_name,ascending=True)
                df.index = df[start_mapped_name]
                df['delta'] = (df[end_mapped_name]-df[start_mapped_name]).dt.seconds.fillna(np.float64(0))
                zero_interval = df[df['delta']==0].index
                df.drop(zero_interval, inplace=True)
                total_counts = df['delta'].values
                plot_acf(total_counts, lags=10, title="{} Autocorrelation for Elasped time between {}-{} : Hours {}".format(
                                                                                                           self.name,start_event, 
                                                                                                           end_event, "Overall"))
                t = acf(total_counts, nlags=10, fft=False)
                print("Overall Avg")
                print(np.mean(t[1:]))
                avg = []
                for x in self.hour_ranges:
                    hour_slice = df.between_time(*x)  
                    total_counts = hour_slice['delta'].values
                    plot_acf(total_counts, lags=10, title="{} Autocorrelation for Elasped time between {}-{} : Hours {}".format(
                                                                                                           self.name,start_event, 
                                                                                                           end_event, x))
                    t = acf(total_counts, nlags=10, fft=False)
                    avg.append(np.mean(t[1:]))
                print("Hourly Avg")
                print(np.mean(avg))

    def calculate_avg_arrival_rates_per_hour(self):
        hour_stats = {event_name:"Not Recorded" for event_name in self.event_names}
        for event_name in tqdm_notebook(self.event_names):
            mapped_name = self.source_map[event_name]
            if mapped_name is not None:
                all_columns = [self.unique_id, mapped_name]
                df = dd.read_csv(self.data_path, usecols=all_columns)
                df = df.dropna()
                df = df.compute()
                df[mapped_name] = pd.to_datetime(df[mapped_name], errors="coerce", infer_datetime_format=True)
                df.index = df[mapped_name]
                total_counts = df.groupby([df.index.year, df.index.month, df.index.day, df.index.hour])[self.unique_id].count().values
                fig, ax = plt.subplots()
                x = [e for e in range(0,24*10)]
                x = np.array(x)
                days = np.random.choice(360, 10, replace=False)
                total = []
                [total.extend(total_counts[(12*x):(12*x)+24]) for x in days]
                c = np.mean(total)
                fs,pw = ss.periodogram(x)
                max_y = max(pw)  # Find the maximum y value
                dom_freq = fs[pw.argmax()]
                amp = sqrt(sum(n*n for n in total)/len(total)) * sqrt(2) 
                #print(dom_freq)
                #print(amp)
                #print(c)
                #print(fs)                 
                #ax.plot(fs,pw)
                params, params_covariance = optimize.curve_fit(sin_func, x, total, p0=[amp, c])
                ax.plot(x,total, 'bo')
                ax.plot(x, sin_func(x, params[0], params[1]), label='Fitted function')
                hour_avgs = {}
                for x in self.hour_ranges:
                    hour_slice = df.between_time(*x)  
                    total_counts = hour_slice.groupby([hour_slice.index.year, hour_slice.index.month, hour_slice.index.day, hour_slice.                                                       index.hour])[self.unique_id].count().values
                    mean = np.mean(total_counts)
                    hour_avgs[x] = mean
                hour_stats[event_name] = hour_avgs

        for k,v in hour_stats.items():
            x = list(v.keys())
            x_vals = np.array([e for e in range(0,24)])
            y = list(v.values())
            fig, ax = plt.subplots()
            ax.plot(x_vals, y, 'bo')
            ax.plot(x_vals, sin_func(x_vals, params[0], params[1]), label='Fitted function')

        return hour_stats 

    def show_number_of_arrivals_per_minute_graph(self):
        for event_name in tqdm_notebook(self.event_names):
            mapped_name = self.source_map[event_name]
            if mapped_name is not None:
                all_columns = [self.unique_id, mapped_name]
                df = dd.read_csv(self.data_path, usecols=all_columns)
                df = df.dropna()
                df = df.compute()
                df[mapped_name] = pd.to_datetime(df[mapped_name], errors="coerce", infer_datetime_format=True)
                df.index = df[mapped_name]
                total_counts = df.groupby([df.index.year, df.index.month, df.index.day, df.index.hour, df.index.minute])[self.unique_id].count().values
                fig, ax = plt.subplots()
                x = np.linspace(0, len(total_counts), len(total_counts))
                ax.plot(x, total_counts)   

    def show_number_of_arrivals_per_hour_graph(self):
        for event_name in tqdm_notebook(self.event_names):
            mapped_name = self.source_map[event_name]
            if mapped_name is not None:
                all_columns = [self.unique_id, mapped_name]
                df = dd.read_csv(self.data_path, usecols=all_columns)
                df = df.dropna()
                df = df.compute()
                df[mapped_name] = pd.to_datetime(df[mapped_name], errors="coerce", infer_datetime_format=True)
                df.index = df[mapped_name]
                total_counts = df.groupby([df.index.year, df.index.month, df.index.day, df.index.hour])[self.unique_id].count().values
                fig, ax = plt.subplots()
                x = np.linspace(0, len(total_counts), len(total_counts))
                ax.plot(x, total_counts)   

    def show_avg_number_of_arrivals_per_hour_graph(self):
        for event_name in tqdm_notebook(self.event_names):
            mapped_name = self.source_map[event_name]
            if mapped_name is not None:
                all_columns = [self.unique_id, mapped_name]
                df = dd.read_csv(self.data_path, usecols=all_columns)
                df = df.dropna()
                df = df.compute()
                df[mapped_name] = pd.to_datetime(df[mapped_name], errors="coerce", infer_datetime_format=True)
                df.index = df[mapped_name]
                hour_groups = df.groupby([df.index.year, df.index.month, df.index.day, df.index.hour])
                rec = {x:{"total":0, "obvs":0} for x in range(0,24)}
                for name, group in hour_groups:
                    x = name[3]
                    rec[x]["total"] += group[self.unique_id].count()
                    rec[x]["obvs"] += 1
                
                means = [rec[x]["total"]/rec[x]["obvs"] for x in range(0,24)]
                x = [x for x in range(0,24)]
                fig, ax = plt.subplots()
                print(self.name, event_name)
                print("Min Hour")
                print(np.argmin(means))
                print("Max Hour")
                print(np.argmax(means)) 
                ax.plot(x,means)
                ax.set_xlabel("Hours (0-23)")
                ax.set_ylabel("Average Event Count")
                plt.show()
                     
    def show_avg_number_of_arrivals_per_minute_graph(self):
        for event_name in tqdm_notebook(self.event_names):
            mapped_name = self.source_map[event_name]
            if mapped_name is not None:
                all_columns = [self.unique_id, mapped_name]
                df = dd.read_csv(self.data_path, usecols=all_columns)
                df = df.dropna()
                df = df.compute()
                df[mapped_name] = pd.to_datetime(df[mapped_name], errors="coerce", infer_datetime_format=True)
                df.index = df[mapped_name]
                minute_groups = df.groupby([df.index.year, df.index.month, df.index.day, df.index.hour, df.index.minute])
                rec = {x:{"total":0, "obvs":0} for x in range(0,60)}
                for name, group in minute_groups:
                    x = name[4]
                    rec[x]["total"] += group[self.unique_id].count()
                    rec[x]["obvs"] += 1
                
                means = [rec[x]["total"]/rec[x]["obvs"] for x in range(0,60)]
                x = [x for x in range(0,60)]
                fig, ax = plt.subplots()
                print(self.name, event_name)
                ax.plot(x,means) 
                ax.set_xlabel("Minutes (0-59)")
                ax.set_ylabel("Average Event Count")
                plt.show()

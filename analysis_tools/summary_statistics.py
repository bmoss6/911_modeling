import csv
import pandas as pd
import dask.dataframe as dd
import scipy.stats as st
import numpy as np
import json
import itertools
import os.path
from tqdm import tqdm_notebook

class Summary_Statistics():
    def __init__(self, data_name, data_path, data_map_path="/home/blakemoss/911_modeling/ts_data_map.csv"):
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
        if self.source_map['Dtype'] is not None:
            self.dtype = json.loads(self.source_map['Dtype'])
        else:
            self.dtype = None
        self.num_days = None
 
    def calculate_start_end_dates(self):
        inc_closed_field = self.source_map['incident_closed']
        all_columns = [inc_closed_field]
        df = dd.read_csv(self.data_path, usecols=all_columns)
        df = df.dropna()
        df = df.compute()
        dates = pd.to_datetime(df[inc_closed_field], errors="coerce", infer_datetime_format=True)
        return dates.min().strftime("%b %Y"), dates.max().strftime("%b %Y")
 
    def calculate_unique_events(self):
        all_columns = [self.unique_id]
        df = dd.read_csv(self.data_path, usecols=all_columns)
        df = df.dropna()
        df = df.compute()
        return len(df[self.unique_id].unique())

    def calculate_unique_incident_types_and_counts(self):
        inc_type_stats = {}
        if self.call_type_field is not None and self.priority_type_field is not None:
            inc_closed_field = self.source_map['incident_closed']
            all_columns = [self.unique_id, self.call_type_field, self.priority_type_field, inc_closed_field]
            if self.dtype is not None:
                df = dd.read_csv(self.data_path, usecols=all_columns, dtype=self.dtype)
            else:
                df = dd.read_csv(self.data_path, usecols=all_columns)
            df = df.dropna()
            df = df.compute()
            stats = {}
            different_call_types = df[self.call_type_field].unique()
            different_priority_types = len(df[self.priority_type_field].unique())
            df[inc_closed_field] = pd.to_datetime(df[inc_closed_field], errors="coerce", infer_datetime_format=True)
            df.index = df[inc_closed_field]
            num_days = len(df.index.normalize().unique())
            for call_type in tqdm_notebook(different_call_types):
                filtered = len(df[df[self.call_type_field]== call_type][self.unique_id].unique())
                per_day_avg = filtered/num_days
                interval_stats = self.calculate_average_event_intervals_call_type(call_type)
                stats[call_type] = {"day_avg": per_day_avg, "interval_stats":interval_stats} 
            return stats, different_priority_types 
        else:
            return None, None 

    def calculate_average_events_per_day(self):
        day_stats = {event_name:"Not Recorded" for event_name in self.event_names}
        for event_name in tqdm_notebook(self.event_names):
            mapped_name = self.source_map[event_name]
            if mapped_name is not None:
                all_columns = [self.unique_id, mapped_name]
                df = dd.read_csv(self.data_path, usecols=all_columns)
                df = df.dropna()
                df = df.compute()
                df[mapped_name] = pd.to_datetime(df[mapped_name], errors="coerce", infer_datetime_format=True)
                df.index = df[mapped_name]
                num_days = len(df.index.normalize().unique())
                per_day_avg = len(df[self.unique_id].unique())/num_days
                day_stats[event_name] = per_day_avg
        return day_stats 

    def calculate_average_event_intervals_call_type(self, call_type):
        interval_stats = {"{}-{}".format(start, end):"Not Recorded" for start, end in self.intervals}
        for start_event, end_event in self.intervals:
            start_mapped_name = self.source_map[start_event]
            end_mapped_name = self.source_map[end_event]
            if start_mapped_name is not None and end_mapped_name is not None:
                all_columns = [self.unique_id, self.call_type_field, start_mapped_name, end_mapped_name]
                df = dd.read_csv(self.data_path, usecols=all_columns)
                df = df.dropna()
                df = df.compute()
                df[start_mapped_name] = pd.to_datetime(df[start_mapped_name], errors="coerce", 
                                                             infer_datetime_format=True)
                df[end_mapped_name] = pd.to_datetime(df[end_mapped_name], errors="coerce", infer_datetime_format=True)
                df['delta'] = (df[end_mapped_name]-df[start_mapped_name]).dt.seconds.fillna(np.float64(0))
                zero_interval = df[df['delta']==0].index
                df.drop(zero_interval, inplace=True)
                filtered = df[df[self.call_type_field]== call_type]
                #df = df.sort_values(by=mapped_name,ascending=True)
                #df.index = df[mapped_name]
                total_counts = filtered['delta'].values
                avg = np.sum(total_counts)/len(total_counts)
                interval_stats["{}-{}".format(start_event, end_event)] = avg
        return interval_stats 

    def calculate_average_event_intervals(self):
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
                #df = df.sort_values(by=mapped_name,ascending=True)
                #df.index = df[mapped_name]
                df['delta'] = (df[end_mapped_name]-df[start_mapped_name]).dt.seconds.fillna(np.float64(0))
                zero_interval = df[df['delta']==0].index
                df.drop(zero_interval, inplace=True)
                total_counts = df['delta'].values
                avg = np.sum(total_counts)/len(total_counts)
                interval_stats["{}-{}".format(start_event, end_event)] = avg
        return interval_stats

    def write_statistics(self, stats_file="/home/blakemoss/911_modeling/summary_stats.csv"):
        avg_events = self.calculate_average_events_per_day()
        interval_events = self.calculate_average_event_intervals()
        uniq_incs, priorities = self.calculate_unique_incident_types_and_counts()
        start, end = self.calculate_start_end_dates()
        uniq_events = self.calculate_unique_events()
        data_set = self.source_map['Dataset']
        service_type = self.source_map['Type']
        date_range = "{}-{}".format(start, end)
        call_rec = avg_events['call_received']
        entrd_cad = avg_events['entered_to_cad']
        first_unit_asgnd = avg_events['first_unit_assigned']
        first_unit_enrt = avg_events['first_unit_enroute']
        first_unit_onscene = avg_events['first_unit_onscene']
        first_unit_tohospital = avg_events['first_unit_tohospital']
        first_unit_athospital = avg_events['first_unit_athospital']
        inc_closed = avg_events['incident_closed']
        call_recv_to_entered = interval_events["call_received-entered_to_cad"]
        entrd_asgnd = interval_events["entered_to_cad-first_unit_assigned"] 
        call_rcv_to_asgnd = interval_events["call_received-first_unit_assigned"]
        unit_asgnd_enrt = interval_events["first_unit_assigned-first_unit_enroute"] 
        enrt_onscene = interval_events["first_unit_enroute-first_unit_onscene"]
        tohos_athos = interval_events["first_unit_tohospital-first_unit_athospital"]
        at_hos_inc_closed = interval_events["first_unit_athospital-incident_closed"]
        on_scn_closed = interval_events["first_unit_onscene-incident_closed"]

        if uniq_incs is not None:
            uniq_types = len(uniq_incs.keys())
        else:
            uniq_types = "Classifications not Recorded"

        row = [data_set, service_type, date_range, uniq_events, call_rec, entrd_cad, first_unit_asgnd, 
               first_unit_enrt, first_unit_onscene, first_unit_tohospital, first_unit_athospital, inc_closed,
               call_recv_to_entered, entrd_asgnd, call_rcv_to_asgnd, unit_asgnd_enrt, enrt_onscene, tohos_athos,
               at_hos_inc_closed, on_scn_closed, uniq_types, priorities]
        new_file = os.path.isfile(stats_file) 

        with open(stats_file, 'a+') as f:
            writer = csv.writer(f)
            if new_file is False:
                first_row = ["Dataset", "Type", "Date Range", "Number of Unique Non Officer-Initiated Incidents (Calls)",
                         "Avg Call Received Event Per Day", "Avg Incident Entered into CAD Events Per Day", 
                         "Avg First Unit Assigned to Dispatch Events Per Day", "Avg First Unit Enroute to Incident Events Per Day", 
                         "Avg First Unit Arrives on Scene Events Per Day", "Avg First Unit Enroute to Hospital Events Per Day", 
                         "Avg First Unit Arrives at Hospital Events Per Day", "Avg Incident Closed Events Per Day", 
                         "Avg Elasped Time (Seconds) Call Received to Call Entered into CAD", 
                         "Avg Elapsed Time (Seconds) Call Entered into CAD to First Unit Assigned to Dispatch",
                         "Avg Elapsed Time (Seconds) Call Received to First Unit Assigned to Dispatch", 
                         "Avg Elapsed Time (Seconds) First Unit Assigned to Dispatch to First Unit Enroute", 
                         "Avg Elapsed Time (Seconds) First Unit Enroute to First Unit Onscene (Travel Time)", 
                         "Avg Elapsed Time (Seconds) First Unit Onscene to First Unit Leave to Hospital", 
                         "Avg Elapsed Time (Seconds) First Unit Leave to Hospital to Incident Closed", 
                         "Avg Elapsed Time (Seconds) First Unit Onscene to Incident Closed", 
                         "Number of Unique Call Types", "Number of Unique Priority Codes"]
                writer.writerow(first_row)
            writer.writerow(row)

        if uniq_incs is not None:
            type_count_file_name = "{}_call_type_counts.csv".format(data_set.replace(".csv",""))
            with open(type_count_file_name, 'w') as f:
                writer = csv.writer(f)
                row = ["Call Type", "Avg Occurence Per Day", "Avg Elasped Time (Seconds) Call Received to Call Entered into CAD", 
                "Avg Elapsed Time (Seconds) Call Entered into CAD to First Unit Assigned to Dispatch",
                "Avg Elapsed Time (Seconds) Call Received to First Unit Assigned to Dispatch", 
                "Avg Elapsed Time (Seconds) First Unit Assigned to Dispatch to First Unit Enroute", 
                "Avg Elapsed Time (Seconds) First Unit Enroute to First Unit Onscene (Travel Time)", 
                "Avg Elapsed Time (Seconds) First Unit Onscene to First Unit Leave to Hospital", 
                "Avg Elapsed Time (Seconds) First Unit Leave to Hospital to Incident Closed", 
                "Avg Elapsed Time (Seconds) First Unit Onscene to Incident Closed"] 
                writer.writerow(row)
                for typ_name,obj in uniq_incs.items():
                    call_recv_to_entered = obj["interval_stats"]["call_received-entered_to_cad"]
                    entrd_asgnd = obj["interval_stats"]["entered_to_cad-first_unit_assigned"] 
                    call_rcv_to_asgnd = obj["interval_stats"]["call_received-first_unit_assigned"]
                    unit_asgnd_enrt = obj["interval_stats"]["first_unit_assigned-first_unit_enroute"] 
                    enrt_onscene = obj["interval_stats"]["first_unit_enroute-first_unit_onscene"]
                    tohos_athos = obj["interval_stats"]["first_unit_tohospital-first_unit_athospital"]
                    at_hos_inc_closed = obj["interval_stats"]["first_unit_athospital-incident_closed"]
                    on_scn_closed = obj["interval_stats"]["first_unit_onscene-incident_closed"]
                    day_avg = obj["day_avg"]
                    row = [typ_name, day_avg, call_recv_to_entered, entrd_asgnd, call_rcv_to_asgnd, 
                           unit_asgnd_enrt, enrt_onscene, tohos_athos, at_hos_inc_closed, on_scn_closed]
                    writer.writerow(row)

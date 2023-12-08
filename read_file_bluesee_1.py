# general imports
import os
import binascii
import datetime
from struct import unpack
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np

def decode_logged_file(directory, condition, demo = False):
    '''
    This method is used to read the logged file from the BlueSee application.
    
    Input arguments:
    - directory (String) - name of the directory in which logged files are located.
    - condition (String) - condition of the data collection trial. The 4 conditions are neck_bad, back_bad, neck_goood, and back_good (See above markdown)
    - demo (Boolean) - whether the logged file is for the real-time demo application or not.
    
    Returns:
    - d (Dictionary) - dictionary is returned where the key is the timestamp of the computer and the value is the logged hex data from the sensor
    '''
    d = {}
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        if os.path.isfile(file):
            if condition in filename:
                with open(file, 'r') as f:
                    lines = [line.rstrip() for line in f]
                    for l in lines:
                        if 'received update from characteristic CF54BF43-3D66-4666-8FD3-7DF5788B73C1___E8A68B2A-B616-45C0-B8D0-D9DDF447731E' in l:
                            time = l[:13]
                            data = l[-45:-1]
                            if demo:
                                d[time] = data.replace(" ", "")
                            else:
                                d[time] = (data.replace(" ", ""), filename[:-4]) # filename includes the condition and trial number information (condition_trialnum)
    return d

def decode_hex_value(cond_dict, demo = False):
    '''
    This method is used to decode the hex value from the logged data of the sensors. The hex value is deciphered according to the byte information for each element provided by Hocoma.
    
    Input arguments:
    - cond_dict (Dictionary) - the condition dictionary from the output of the decode_logged_file method.
    - demo (Boolean) - whether the logged file is for the real-time demo application or not.
    
    Returns:
    - l (List) - list of tuples is returned that contains the timestamp of the computer, quaternion components (W, X, Y, Z), timestamp of the sensor, sensor status, and label (condition_trialnum)
    '''
    l = []
    for key, value in cond_dict.items():
        if demo:
            pckt = binascii.unhexlify(value)
        else:
            pckt = binascii.unhexlify(value[0])
        # quaternion components (W, X, Y, Z) (32 bits each)
        W = unpack("<f", pckt[0:4])[0] 
        X = unpack("<f", pckt[4:8])[0] 
        Y = unpack("<f", pckt[8:12])[0] 
        Z = unpack("<f", pckt[12:16])[0]
        # sensor timestamp (24 bits)
        TS = int.from_bytes(pckt[16:19], "little") 
        TS = datetime.datetime.fromtimestamp(TS / 1e6) # for microsecond precision
        # sensor status (8 bits)
        S = ''.join(format(byte, '08b')[::-1] for byte in pckt[19:20]) 
        if demo:
            t = (key, (W, X, Y, Z), TS, S)
        else:
            L = value[1] # label
            t = (key, (W, X, Y, Z), TS, S, L)
        l.append(t)
    return l

def create_dataframe(cond_list, demo = False):
    '''
    This method is used to create a dataframe from the output of the logged file that was decoded.
    
    Input arguments:
    - cond_list (List) - list of tuples of all of the individual elements in the logged data (output of decode_hex_values)
    - demo (Boolean) - whether the logged file is for the real-time demo application or not.
    
    Returns:
    - df (Pandas Dataframe) - dataframe of all of the logged data elements
    '''
    if demo:
        df = pd.DataFrame(data = cond_list, 
                      index = range(0, len(cond_list)),
                      columns = ['Notebook_Timestamp','Quaternion_Orientation_(W, X, Y, Z)','Sensor_Timestamp','Status'])
    else:
        df = pd.DataFrame(data = cond_list, 
                      index = range(0, len(cond_list)),
                      columns = ['Notebook_Timestamp','Quaternion_Orientation_(W, X, Y, Z)','Sensor_Timestamp','Status','Label'])
    df["Notebook_Timestamp"] = pd.to_datetime(df["Notebook_Timestamp"], format='%H:%M:%S.%f')
    return df

def check_sensor_orientation(df):
    '''
    This method is used to make sure that the sensors are in the correct orientation (i.e., not upside down).
    
    Input arguments:
    - df (Pandas Dataframe) - the dataframe containing the quaternion orientation information to check.
    
    Returns:
    - df (Pandas Dataframe) - filtered if there are any datapoints whose orientation is not correct.
    '''
    df_copy = df.copy(deep=True)
    df_copy['W'] = df['Quaternion_Orientation_(W, X, Y, Z)'].str[0].to_numpy()
    df_copy['X'] = df['Quaternion_Orientation_(W, X, Y, Z)'].str[1].to_numpy()
    df_copy['Y'] = df['Quaternion_Orientation_(W, X, Y, Z)'].str[2].to_numpy()
    df_copy['Z'] = df['Quaternion_Orientation_(W, X, Y, Z)'].str[3].to_numpy()
    
    # upside down orientation was tested with test_up.txt and test_down.txt files in the data_collection folder
    # test_up.txt is the data collected in the correct upright orientation.
    # test_down.txt is the data collected in the upside down orientation.
    df = df_copy[~((df_copy['W'].astype(int) > 0) & (df_copy['X'].astype(int) > 0) & (df_copy['Y'].astype(int) > 0) & (df_copy['Z'].astype(int) < 0))]
    df = df.drop(columns=['W', 'X', 'Y', 'Z'])
    return df

def synchronize(neck_df, back_df, demo = False, num_trials = 0, is_good = False):
    '''
    This method is used to synchronize the sensor data according to the global notebook and local sensor timestamps.
    
    Input arguments:
    - neck_df (Pandas Dataframe) - dataframe containing the sensor information when it is placed on the neck
    - back_df (Pandas Dataframe) - dataframe containing the sensor information when it is placed on the back
    - demo (Boolean) - whether the logged file is for the real-time demo application or not.
    - num_trials (Integer) - number of trials of data collection to iterate over
    - is_good (Boolean) - whether the condition is the Good Posture condition.
    
    Returns:
    - df (Pandas Dataframe) - synchronized dataframe of the neck_df and back_df
    '''
    
    if demo:
        
        neck_df, back_df = synchronize_notebook_timestamps(neck_df, back_df)
        df = synchronize_sensor_timestamps(neck_df, back_df)
        
    else:
        
        # iterate over trials
        trials = list(range(1, num_trials + 1))

        is_good_df = []
        all_df = []

        for t in trials:
            
            # get the filtered dataframe for the trial of interest
            if t < 10:
                neck_df_t = neck_df[neck_df['Label'].str.slice(start=-2) == "_{}".format(t)]
                back_df_t = back_df[back_df['Label'].str.slice(start=-2) == "_{}".format(t)]
            else:
                neck_df_t = neck_df[neck_df['Label'].str.slice(start=-2) == str(t)]
                back_df_t = back_df[back_df['Label'].str.slice(start=-2) == str(t)]

            neck_df_t.reset_index(drop=True, inplace=True)
            back_df_t.reset_index(drop=True, inplace=True)
            
            # first synchronize the notebook timestamps
            neck_df_t, back_df_t = synchronize_notebook_timestamps(neck_df_t, back_df_t)

            # then synchronize the sensor timestamps 
            df = synchronize_sensor_timestamps(neck_df_t, back_df_t)

            # in the Good Posture condition, not much movement was made. 
            # Therefore, to get more variation of data, data from several trials are combined
            if is_good:
                if (df.shape[0] >= 250) and (len(is_good_df) != 4): # make sure a trial has 1000 datapoints when concatenated
                    df = df.head(250)
                    is_good_df.append(df)
                    if len(is_good_df) == 4:
                        df = concat_dataframes(is_good_df, sort_col='Notebook_Timestamp_Neck')
                        df['Label_Neck'] = "neck_good_{}".format(t)
                        df['Label_Back'] = "back_good_{}".format(t)
                        df = df.head(1000) # one trial has 1000 datapoints
                        all_df.append(df)
                        is_good_df.clear()
                else:
                    print("trial number {} has insufficient number of datapoints: {}".format(t, df.shape[0]))
            
            # More movement was made during data collection of FHP. Therefore concatenating data among trials was not necessary
            if not is_good:
                if df.shape[0] >= 1000:
                    df = df.head(1000) # one trial has 1000 datapoints
                    all_df.append(df)
                else:
                    print("trial number {} has insufficient number of datapoints: {}".format(t, df.shape[0]))
            
        df = concat_dataframes(all_df, sort_col='Notebook_Timestamp_Neck')

    return df

def synchronize_notebook_timestamps(neck_df, back_df):
    '''
    This method is used to match the first notebook timestamp of the two sensor logged data of the neck and back.
    The dataframes should be the same condition and trial.
    
    Input arguments:
    - neck_df (Pandas Dataframe) - dataframe containing the sensor information when it is placed on the neck
    - back_df (Pandas Dataframe) - dataframe containing the sensor information when it is placed on the back
    
    Returns:
    - neck_df (Pandas Dataframe) - dataframe that is filtered to match the first notebook timestamp
    - back_df (Pandas Dataframe) - dataframe that is filtered to match the first notebook timestamp
    '''
    
    # first notebook timestamp of each dataframe
    neck_nb_ts = neck_df['Notebook_Timestamp'].iloc[0]
    back_nb_ts = back_df['Notebook_Timestamp'].iloc[0]
    
    # if the notebook timestamp starts earlier for the neck dataframe
    if (neck_nb_ts < back_nb_ts):
        # the neck dataframe starting index is delayed
        start_index = neck_df.index[[(abs(neck_df.iloc[:, 0]-back_nb_ts)).idxmin()]].item()
        neck_df = neck_df[neck_df.index >= start_index]
    else:
        start_index = back_df.index[[(abs(back_df.iloc[:, 0]-neck_nb_ts)).idxmin()]].item()
        back_df = back_df[back_df.index >= start_index]
        
    return neck_df, back_df

def synchronize_sensor_timestamps(neck_df, back_df):
    '''
    This method is used to synchronize the sensor timestamps of the two sensor logged data of the neck and back.
    The dataframes should be the same condition and trial, and already matched to the first notebook timestamp.
    
    Input arguments:
    - neck_df (Pandas Dataframe) - dataframe containing the sensor information when it is placed on the neck
    - back_df (Pandas Dataframe) - dataframe containing the sensor information when it is placed on the back
    
    Results:
    - df (Pandas Dataframe) - merged dataframe of the sensor_neck and sensor_back data by synchronizing the sensor timestamps
    '''
    
    neck_s_ts1 = neck_df['Sensor_Timestamp'].iloc[0]
    neck_s_ts2 = neck_df['Sensor_Timestamp'].iloc[1]
    back_s_ts1 = back_df['Sensor_Timestamp'].iloc[0]
    back_s_ts2 = back_df['Sensor_Timestamp'].iloc[1]

    # get the sensor timestamp difference 
    neck_s_ts_diff = neck_s_ts2 - neck_s_ts1
    back_s_ts_diff = back_s_ts2 - back_s_ts1

    neck_df.reset_index(drop=True, inplace=True)
    back_df.reset_index(drop=True, inplace=True)

    # make sure timestamp is increasing at a constant rate with the sensor timestamp difference
    for i in range(1, neck_df.shape[0]):
        prev = neck_df['Sensor_Timestamp'].iloc[i-1]
        if (neck_df['Sensor_Timestamp'].iloc[i] < prev):
            neck_df.loc[i, 'Sensor_Timestamp'] = prev + neck_s_ts_diff

    for i in range(1, back_df.shape[0]):
        prev = back_df['Sensor_Timestamp'].iloc[i-1]
        if (back_df['Sensor_Timestamp'].iloc[i] < prev):
            back_df.loc[i, 'Sensor_Timestamp'] = prev + back_s_ts_diff

    # if the sensor timestamp starts earlier for the neck dataframe
    if (neck_s_ts1 < back_s_ts1):
        # the sensor timestamps of the neck are shifted by the difference between the sensor timestamps of the neck and back dataframes
        s_ts_diff = back_s_ts1 - neck_s_ts1
        neck_df['Sensor_Timestamp'] = neck_df['Sensor_Timestamp'] + s_ts_diff

    else:
        s_ts_diff = neck_s_ts1 - back_s_ts1
        back_df['Sensor_Timestamp'] = back_df['Sensor_Timestamp'] + s_ts_diff

    # set the starting time of the sensor timestamp to 0
    neck_s_ts = neck_df['Sensor_Timestamp'].iloc[0]
    back_s_ts = back_df['Sensor_Timestamp'].iloc[0]
    neck_df['Sensor_Timestamp'] = neck_df['Sensor_Timestamp'] - neck_s_ts
    back_df['Sensor_Timestamp'] = back_df['Sensor_Timestamp'] - back_s_ts
    
    # make sure that the dataframes for the neck and back are of the same shape
    if (neck_df.shape[0] < back_df.shape[0]):
        back_df = back_df[back_df.index < neck_df.shape[0]]
    else:
        neck_df = neck_df[neck_df.index < back_df.shape[0]]

    # merge data on the synchronized sensor timestamps
    back_df = back_df.set_index('Sensor_Timestamp').reindex(neck_df.set_index('Sensor_Timestamp').index, method='nearest').reset_index()
    df = pd.merge(neck_df, back_df, on='Sensor_Timestamp', suffixes=('_Neck', '_Back'))
    df = format_timestamps(df, ['Notebook_Timestamp_Neck', 'Notebook_Timestamp_Back'])
    
    return df

def concat_dataframes(cond_list, sort_col):
    '''
    This method is used to concatenate dataframes in a list.
    
    Input arguments:
    - cond_list (List) - list of dataframes to concatenate
    - sort_col (String) - column of dataframe to sort the data by
    
    Results:
    - df (Pandas Dataframe) - concatenated dataframe
    '''
    df = pd.concat(cond_list)
    df = df.sort_values(by=[sort_col])
    df.reset_index(drop=True, inplace=True)
    return df

def format_timestamps(df, col_name_list):
    '''
    This method is used to format the timestamp data.
    
    Input arguments:
    - df (Pandas Dataframe) - dataframe that contains a timestamp column to format
    - col_name_list (List) - list of columns to format the timestamps
    
    - df (Pandas Dataframe) - dataframe whose timestamp columns are formatted
    '''
    for col in col_name_list:
        df[col] = df[col].dt.strftime("%H:%M:%S.%f")
    return df
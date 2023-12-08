from math import sqrt
import random

def create_df_model(df, demo = False):
    '''
    This method is used to calculate the quaternion difference of the two quaternion sensor data and create dataframe for data visualization and modeling
    
    Input arguments:
    - df (Pandas dataframe) - preprocessed synchronized dataframe to work with
    - demo (Boolean) - whether the logged file is for the real-time demo application or not.
    
    Returns:
    - df (Pandas Dataframe) - dataframe containing the calculated quaternion difference of the two quaternion sensor data.
    '''
    
    df = df.copy(deep=True)
    
    a = 'Quaternion_Orientation_(W, X, Y, Z)_Neck'
    b = 'Quaternion_Orientation_(W, X, Y, Z)_Back'
    
    ### Extract the quaternion components for each quaternion
    # Q1 = Neck
    df['W1'] = df[a].str[0].to_numpy()
    df['X1'] = df[a].str[1].to_numpy()
    df['Y1'] = df[a].str[2].to_numpy()
    df['Z1'] = df[a].str[3].to_numpy()

    # Q2 = Back
    df['W2'] = df[b].str[0].to_numpy()
    df['X2'] = df[b].str[1].to_numpy()
    df['Y2'] = df[b].str[2].to_numpy()
    df['Z2'] = df[b].str[3].to_numpy()

    # Inverse(Q1) - Inverse of a quaternion is equal to the conjugate of the quaternion (source: https://www.3dgep.com/understanding-quaternions/)
    # Conjugate quaternion is calculated by negating the X, Y, Z components (source: https://www.meccanismocomplesso.org/en/hamiltons-quaternions-and-3d-rotation-with-python/)
    df['X1'] = df['X1'] * -1
    df['Y1'] = df['Y1'] * -1
    df['Z1'] = df['Z1'] * -1

    # Quaternion Difference = Q2 * Inverse(Q1)
    # Quaternion multiplication (source: https://www.meccanismocomplesso.org/en/hamiltons-quaternions-and-3d-rotation-with-python/)
    df['W'] = df['W1'] * df['W2'] - df['X1'] * df['X2'] - df['Y1'] * df['Y2'] - df['Z1'] * df['Z2']
    df['X'] = df['W1'] * df['X2'] + df['X1'] * df['W2'] + df['Y1'] * df['Z2'] - df['Z1'] * df['Y2']
    df['Y'] = df['W1'] * df['Y2'] + df['Y1'] * df['W2'] + df['Z1'] * df['X2'] - df['X1'] * df['Z2']
    df['Z'] = df['W1'] * df['Z2'] + df['Z1'] * df['W2'] + df['X1'] * df['Y2'] - df['Y1'] * df['X2']
    
    if demo:
        
        model_variables = ['W', 'X', 'Y', 'Z', 'Notebook_Timestamp_Neck']
        df = df[df.columns.intersection(model_variables)]
        df.columns = df.columns.str.replace('Notebook_Timestamp_Neck', 'Timestamp')
        
    else:
        
        # Label the data for binary classification (1 = good // 0 = bad)
        df['Label'] = np.where(df['Label_Neck'].str.slice(start=5, stop=9) == 'good', 1, 0)
        df['Color'] = np.where(df['Label_Neck'].str.slice(start=5, stop=9) == 'good', 'b', 'r') # for data visualization
        df['Trial'] = np.where(df['Label_Neck'].str.slice(start=-2, stop=-1) == '_', df['Label_Neck'].str.slice(start=-1), df['Label_Neck'].str.slice(start=-2))

        model_variables = ['W', 'X', 'Y', 'Z', 'Label', 'Trial', 'Color']

        df = df[df.columns.intersection(model_variables)]
    
    return df

def split_df_model(df):
    '''
    This method is used to split the input dataframe into the two classes: good posture and FHP
    
    Input arguments:
    - df (Pandas Dataframe) - dataframe to split into the two class dataframes
    
    Returns:
    - df_bad (Pandas Dataframe) - dataframe that contains only the class of FHP data
    - df_good (Pandas Dataframe) - dataframe taht contains only the class of good posture data
    '''
    df_bad = df[df['Label'] == 0]
    df_good = df[df['Label'] == 1]
    df_bad = df_bad.reset_index(drop=True)
    df_good = df_good.reset_index(drop=True)
    return df_bad, df_good

def create_df_model_final(df, demo = False):
    '''
    This method is used to create the final dataframe used for the algorithm implementation, only containing the necessary variables.
    
    Input arugments:
    - df (Pandas Dataframe) - the dataframe that will be used for algorithm implementation
    - demo (Boolean) - whether the logged file is for the real-time demo application or not.
    
    Returns:
    - df (Pandas Dataframe) - Datafarme only containing the necessary information for algorithm implementation
    '''
    if demo:
        model_variables = ['W', 'X', 'Y', 'Z']
    else:
        model_variables = ['W', 'X', 'Y', 'Z', 'Label', 'Trial']
    df = df[df.columns.intersection(model_variables)]
    return df

def train_test_split_trial(df):
    '''
    This method is used to split the data into train/test dataset. 80/20 split is used.
    
    Input arguments:
    - df (Pandas Dataframe) - the data to split the data into train/test datasets
    
    Returns:
    - X_train (Pandas Dataframe) - The X feature vectors used as model input during training (4 features = 4 quaternion difference components)
    - X_test (Pandas Dataframe) - The X feature vecotrs used for testing the model (4 features = 4 quaternion difference components)
    - y_train (Pandas Dataframe) - The ground-truth labels of the data used during training (2 labels = good posture vs FHP)
    - y_test (Pandas Dataframe) - The ground-truth labels of the test data used to calculate model prediction accuracy after training (2 labels = good posture vs FHP)
    '''
    
    df = df.copy(deep=True)
    
    # randomly select test dataset by trial
    # trial is used instead of randomly mixing the dataset to avoid data leakage
    
    df_bad, df_good = split_df_model(df)
    bad_trials = df_bad['Trial'].unique()
    good_trials = df_good['Trial'].unique()
    
    test_trial_good = random.choice(good_trials)
    test_trial_bad = random.choice(bad_trials)

    # obtain the training dataset filtering out the trial number x condition for testing
    df_train_good = df[(df['Trial'] != str(test_trial_good)) & (df['Label'] == 1)]
    df_train_bad = df[(df['Trial'] != str(test_trial_bad)) & (df['Label'] == 0)]

    df_train = concat_dataframes([df_train_good, df_train_bad], sort_col = 'Trial')

    # obtain the testing dataset by filtering for the trial number x condition for testing
    df_test_good = df[(df['Trial'] == str(test_trial_good)) & (df['Label'] == 1)]
    df_test_bad = df[(df['Trial'] == str(test_trial_bad)) & (df['Label'] == 0)]

    df_test = concat_dataframes([df_test_good, df_test_bad], sort_col = 'Trial')

    # shuffle and set X and y variables for both train and test
    df_train = df_train.sample(frac=1).reset_index(drop=True)   
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    
    X_train = df_train.iloc[:, 0:-2]
    y_train = df_train.iloc[:, -2]
    X_test = df_test.iloc[:, 0:-2]
    y_test = df_test.iloc[:, -2]
    
    return X_train, X_test, y_train, y_test
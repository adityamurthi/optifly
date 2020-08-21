# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 10:47:27 2020

@author: aditya
"""


import re
import pandas as pd
from IPython.display import display
from time import time
##Sklearn libraries
from sklearn import base
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split#, GridSearchCV
from sklearn.metrics import r2_score, roc_auc_score, classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
import dill
from datetime import datetime
import datetime as dt
import numpy as np

##Some basic functions  

def time_norm(x):
    "Funtions that normalizes the arrival and departure times."
    y = str(x)
    frac = float(re.sub("[^0-9]", "", y[-2:]))/60.
    stad = str(frac-int(frac))[1:]
    if len(y) <= 2:
        res = str(0)+ stad
    else:##after midnight
        res = y[0:-2]+stad
    return float(res)
    
def get_layover(fldate, t1, t2):
    """Here t2 > t1"""
    t1_str = str(t1)
    if len(t1_str) == 1:
        t1_str = '00:0'+t1_str
    elif len(t1_str) == 2:
        t1_str = '00:'+t1_str
    elif len(t1_str) == 3:
        t1_str = '0' + t1_str[0] + ':' + t1_str[-2:]
    else:
        t1_str = t1_str[0:2] + ':' + t1_str[-2:]
        
    t2_str = str(t2)
    if len(t2_str) == 1:
        t2_str = '00:0'+t2_str
    elif len(t2_str) == 2:
        t2_str = '00:'+t2_str
    elif len(t2_str) == 3:
        t2_str = '0' + t2_str[0] + ':' + t2_str[-2:]
    else:
        t2_str = t2_str[0:2] + ':' + t2_str[-2:]
    ##Combine the date with the time strings to generate a datetime object
    d1 = datetime.strptime(fldate + ' ' + t1_str, "%Y-%m-%d %H:%M")
    d2 = datetime.strptime(fldate + ' ' + t2_str, "%Y-%m-%d %H:%M")
    if d1 < d2:
        diff = d2-d1
        seconds_in_day = 24 * 60 * 60
        minutes = divmod(diff.days * seconds_in_day + diff.seconds, 60)[0]
        return minutes
    else:
        return 60.
        
def total_trip(lat1, del1, lo, lat2, del2):
    """Calculate the total trip time based on the arrival delays for each segment.
    If delay > lo then missed_flight penalty in terms of lo = 3*(lo). If delay < 0, then delay = 0.
    However, right now I just take the negative delay value as is and add it to the total trip time providing
    more slack for the traveller to get to his/her destination on time.
    """
    if del1 > lo:
        lo = 5*lo
    if del1 < 0.:
        del1 = 0.
    return lat1+del1+lo+lat2+del2
    
def func(delay_time,lo):
    if delay_time < 0:
        delay_time = 0.
    if delay_time > lo:
        lo =5*lo
    return delay_time+lo
    
def time_str(x):
    x = str(x)
    if len(x) == 1:
        x_str = '00:0' + x
    elif len(x) == 2:
        x_str = '00:' + x
    elif len(x) == 3:
        x_str = '0' + x[0] +':' + x[-2:]
    else:
        x_str = x[0:2] + ':' + x[-2:]
    return x_str
    
def get_ett(fldate, t1, delay):
    """Compute the estimated trip time, given departure time and date and delay minutes."""
    t1_str = str(t1)
    if len(t1_str) == 1:
        t1_str = '00:0'+t1_str
    elif len(t1_str) == 2:
        t1_str = '00:'+t1_str
    elif len(t1_str) == 3:
        t1_str = '0' + t1_str[0] + ':' + t1_str[-2:]
    else:
        t1_str = t1_str[0:2] + ':' + t1_str[-2:]
    
    ##Create a datetime object
    d1 = dt.datetime.strptime(fldate + ' ' + t1_str, "%Y-%m-%d %H:%M")
    d2 = d1 + dt.timedelta(minutes=delay)
    if d2.day - d1.day == 1:
        dstr = d2.strftime("%H:%M") + '+' + str(d2.day-d1.day) + ' day'
    else:
        dstr = d2.strftime("%H:%M")
    return dstr

def run_optimizer(dfpred, dfopt, time=True, cost=False):
    """Function calculates the most optimal time to fly and provides top n route options """
    colsr = ['fl_date', 'origin', 'dest', 'op_unique_carrier', 'crs_dep_time', 'crs_arr_time', 
         'odp', 'delay_prob', 'predicted_arr_delay(minutes)', 'op_carrier_fl_num', 'carrier_name'] 
    cols1 =['fl_date', 'source', 'connecting_dest', 'carrier_org', 'origin_dep_time',
            'connecting_arr_time', 'leg1', 'leg1_at']
    cols2 = ['fl_date','connecting_origin', 'destination', 'carrier_dest', 'connecting_dep_time', 
             'dest_arr_time', 'leg2', 'leg2_at']

    dfopt1 = (dfopt[cols1].merge(dfpred[colsr], left_on=cols1[0:-1], right_on=colsr[0:7], how='left')
              .drop(colsr[1:7], axis=1)
              .rename(columns={'delay_prob': 'delay_prob_leg1', 
                              'predicted_arr_delay(minutes)': 'pred_arr_delay_leg1(minutes)',
                              'op_carrier_fl_num': 'op_carrier_fl_num_leg1',
                              'carrier_name': 'carrier_name_leg1'}))
    
    #display(dfopt1)
    dfopt2 = (dfopt[cols2].merge(dfpred[colsr], left_on=cols2[0:-1], right_on=colsr[0:7], how='left')
              .drop(colsr[0:7], axis=1)
              .rename(columns={'delay_prob': 'delay_prob_leg2', 
                              'predicted_arr_delay(minutes)': 'pred_arr_delay_leg2(minutes)',
                              'op_carrier_fl_num': 'op_carrier_fl_num_leg2', 'carrier_name': 'carrier_name_leg2'}))
    #display(dfopt2)
    dfoptnew = pd.concat([dfopt1, dfopt2], axis=1)
    #display(dfoptnew.columns)
    
    ##Get layover time.
    dfoptnew['layover_connecting'] = (dfoptnew[['fl_date', 'connecting_arr_time', 'connecting_dep_time']]
                                      .apply(lambda x: get_layover(*x), axis=1))
    
    ##Calculate total time of individual segments.
    #     dfoptnew['estimated_time_leg1(minutes)'] = (dfoptnew['leg1_at'] 
#                                          + dfoptnew[['pred_arr_delay_leg1(minutes)', 'layover_connecting']]
#                                                 .apply(lambda x:func(*x), axis=1))
    dfoptnew['estimated_time_leg1(minutes)'] = dfoptnew['leg1_at'] + dfoptnew['pred_arr_delay_leg1(minutes)']                       
    dfoptnew['estimated_arr_time_leg1 (HH:MM)'] = (dfoptnew[['fl_date', 
                                                            'connecting_arr_time', 
                                                            'pred_arr_delay_leg1(minutes)']]
                                                   .apply(lambda x: get_ett(*x), axis=1))
    
    dfoptnew['estimated_time_leg2(minutes)'] = (dfoptnew['leg2_at'] + dfoptnew['pred_arr_delay_leg2(minutes)'])
    
    dfoptnew['estimated_arr_time_leg2 (HH:MM)'] = (dfoptnew[['fl_date', 
                                                            'dest_arr_time', 
                                                            'pred_arr_delay_leg2(minutes)']]
                                                   .apply(lambda x: get_ett(*x), axis=1))
    colsf = ['leg1_at', 'pred_arr_delay_leg1(minutes)', 'layover_connecting', 'leg2_at', 'pred_arr_delay_leg2(minutes)']
    dfoptnew['final_trip_time'] = dfoptnew[colsf].apply(lambda x: total_trip(*x), axis=1)
    #dfoptnew['final_trip_time'] = dfoptnew['estimated_time_leg1(minutes)'] + dfoptnew['estimated_time_leg2(minutes)']
    dfoptnew['average_delay_probability'] = 0.5*(dfoptnew['delay_prob_leg1']+dfoptnew['delay_prob_leg2'])
    dfoptnew = dfoptnew.sort_values(by='final_trip_time')
    
    return dfoptnew

def feature_gen(df, colsp, pipe=None, train=True, cong=True, rneg=False):
    """
    As a first step we shuffle the data and reset the index of the resulting dataframe 
    so that we can use the dataframe's index to get back the relevant itineraries. This 
    new dataframe will be a subset of the train_0 data table that we will delete to save 
    memory.
    This is a method that takes a raw dataframe containing the relevant 
    flight data (and occupancy and weather data) as input and generates features in 
    a step wise fashion.
    Input:
    df: Train or test dataframe
    colsp: List of pertinent columns
    Optional:
    pipe: a list containing the pipe estimators from fitting the training data.
    train (Boolean): Flag indicating whether training or test
    """
    
    ##Step 1: The pipeline for time norm
    col_times = ['crs_dep_time', 'crs_arr_time']
    if train:
        pipe_time = Pipeline([('cst_tn', ColumnSelectTransformer(col_times)),
                     ('time_norm', TimeGenerator(col_times)),
                    ('time_norm_sc', MinMaxScaler())])
        X_times_arr = pipe_time.fit_transform(df)
    else:##if test
        X_times_arr = pipe[0].transform(df)
    
    X_times = pd.DataFrame(X_times_arr, columns=['sch_dep_time', 'sch_arr_time'])
    #display(X_times)
    #exit(0)
    ##Step2: The pipeline for the Congestion transformer
    if cong:
        col_cong = ['fl_date','op_unique_carrier','origin','dep_time_blk','dest','arr_time_blk']
        if train:
            pipe_cong = Pipeline([('cst_cong', ColumnSelectTransformer(col_cong)),
                             ('cong', CongestionTransformer())])
            X_cong = pipe_cong.fit_transform(df)
        else:
            X_cong = pipe[1].transform(df)
        
        #display(X_cong)
        ##Merge with original dataframe
        df = df.merge(X_cong, left_on=['origin', 'fl_date', 'dep_time_blk'], right_on=['code', 'fl_date', 'time_blk'], how='left')
        ##If there are any NaN rows in congestion, fill with mean
        #df = df.dropna().reset_index().drop('index')

        df = df.drop(['code', 'time_blk'], axis=1)
    
    #display(df)
    #print("Is null", df['congestion'].isnull().any())
    ##Step3:The categorical features need to be OHE and numerical features need to be scaled to a given range (0,1)
    ##The cat feature columns are 
    ##day_of_week, day_of_month, op_unique_carrier. I can use 
    ##sklearn's one-hot-encoder with the ColumnTransformer to do this simulataneously.
    ##For the numerical features: air_time, distance, dep_delay and congestion, I will
    ##use the min_max scaler to scale these values to the range 0 to 1.
    ##Categorical features:
    df['month'] = df['month'].astype(str)
    df['day_of_week'] = df['day_of_week'].astype(str)
    transformer_name = 'ohe'
    columns_to_encode = ['month','day_of_week','op_unique_carrier']
    if train:
        transformer = OneHotEncoder(sparse=False)
        ohe_final = ColumnTransformer([
            (transformer_name, transformer, columns_to_encode)], 
            remainder='passthrough')
        ##Build the pipeline
        pipe_cat = Pipeline([('cst_cat',ColumnSelectTransformer(columns_to_encode)),
                        ('ohe', ohe_final)])
        X_cat = pipe_cat.fit_transform(df)
        cols_cat = list(pipe_cat.named_steps['ohe'].named_transformers_[transformer_name].get_feature_names(columns_to_encode))
    else:
        X_cat = pipe[2].transform(df)
        cols_cat = list(pipe[2].named_steps['ohe'].named_transformers_[transformer_name].get_feature_names(columns_to_encode))
    ##The dataframe containing the transformed categories.
    Xcat = pd.DataFrame(X_cat, columns=cols_cat)
    #display(Xcat)
    
    ##Step 4: Scale the numerical features.
    ##Use the min-max scaler to scale the numerical features air_time, distance and departure_delay
    cols_scale = ['air_time', 'dep_delay', 'distance', 'congestion']
    if rneg:
        mask = df['dep_delay'] < 0.
        df.loc[mask, 'dep_delay'] = 0.
    
    if train:
        transformer_name1 = 'minmax'
        scaler = MinMaxScaler()
        sc_final = ColumnTransformer([
            (transformer_name1, scaler, cols_scale)], 
            remainder='passthrough') 
        ##Build the pipeline
        pipe_num = Pipeline([('cst_num', ColumnSelectTransformer(cols_scale)), 
                             ('scaled_num', sc_final)])
        X_sc = pipe_num.fit_transform(df)
    else:
        X_sc = pipe[3].transform(df)
        
    Xsc = pd.DataFrame(X_sc, columns=cols_scale)
    #display(Xsc)
    ##Merge into a single dataframe 
    colsp = colsp +  ['arr_delay', 'arv_delay']
    Xdf = pd.concat([df[colsp], X_times, Xsc, Xcat], axis=1)
    
    del df
    
    if train:
        if cong:
            return Xdf, [pipe_time, pipe_cong, pipe_cat, pipe_num]
        else:
            return Xdf, [pipe_time, pipe_cat, pipe_num]
    else:##Test data to make predictions on.
        return Xdf
    
def train(conn, balance=False, rneg=False):
    
    ##Pertinent columns to join with the predictions obtained from the
        ##ML model.
    colsp = ['fl_date', 'origin', 'origin_city_name', 'op_unique_carrier', 
             'op_carrier_fl_num', 'crs_dep_time', 'dest', 'dest_city_name',
            'crs_arr_time', 'odp']
    
    print("Reading the training data from the database")
    df = pd.read_sql("select * from training", conn)
    ##Shuffle the dataframe first and pick a small subset of the original dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.dropna().reset_index().drop('index',axis=1)

    ##Generate the feature dataframe
    Xdf, pipe_list = feature_gen(df, colsp, train=True, cong=True, rneg=False)
    names = ['pipe_time.dill', 'pipe_cat.dill', 'pipe_num.dill']
    if len(pipe_list) == 4:
        names = ["pipe_time.dill", "pipe_cong.dill", "pipe_cat.dill", "pipe_num.dill"]
    ##Write these pipe estimators to pickle files
    for p in range(len(pipe_list)):
        print("Writing {} into a dill object!".format(names[p].split('.')[0]))
        dill.dump(pipe_list[p], open(names[p], "wb"))

    
    ##For regression, the target variable has negative values (early flight arrivals) as well. 
    ##If rneg = True, then I will set all the values less than 0 to be zero
    if rneg:
        mask = Xdf['arr_delay'] < 0.0
        Xdf.loc[mask, 'arr_delay'] = 0.0
    #Save the dataframe to a pickle for later use
    #Xdf.to_pickle('train_delay_2018.pkl')
    ##Split into training and test sets
    X_train_df, X_test_df = train_test_split(Xdf, test_size=0.15, stratify=Xdf['arv_delay'].values, random_state=0)

    ##Save memory
    del Xdf

    ##Check if any nulls in the training
    X_train_df = X_train_df.dropna().reset_index().drop('index', axis=1)
    X_test_df = X_test_df.dropna().reset_index().drop('index', axis=1)

    ##Class imbalance handling on the training set. (Why only on the training and not on the entire dataset)
    #if balance:
        ##Use random undersampling to balance the dataset.

    ##Declare the target variables for the classification and regression tasks
    ##For classification the target is arv_delay (0,1)
    ytrain_c = X_train_df['arv_delay'].values
    ytest_c  = X_test_df['arv_delay'].values

    #Xdf = Xdf.drop(['arv_delay', 'arr_delay'], axis=1)
    ##For regression, the target is arr_delay
    ytrain_r = X_train_df['arr_delay'].values
    ytest_r  = X_test_df['arr_delay'].values

    ##Drop the target columns from the two dataframes.
    X_train_df = X_train_df.drop(['arr_delay', 'arv_delay'], axis=1)
    X_test_df = X_test_df.drop(['arr_delay', 'arv_delay'], axis=1)

    ##Now get the flight data for train and test, To be used for the network graph optimization model.
    #fl_train = X_train_df[colsp]
    fl_test  = X_test_df[colsp]

    ##Now get the feature matrix
    colsf = [c for c in list(X_train_df.columns) if c not in colsp]
    #print(colsf)
    train = X_train_df[colsf].to_numpy()
    display(train.shape)
    test  = X_test_df[colsf].to_numpy() 
    display(test.shape)

    print("Predicting using the classification and regression models \n")
    ##Feed to modeling framework
    ##RF and Gradient Boosted trees. 
    #param_grid_c = {'max_depth': range(20, 26)}
    #est1 = GridSearchCV(RandomForestClassifier(n_estimators=300, criterion='gini'), 
    #                   param_grid=param_grid_c, n_jobs=-1, cv=5, scoring='f1_weighted')
    est1 = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=25, n_jobs=-1, random_state=42)
    #param_grid_r = {'alpha': np.linspace(1e-3, 10, 20)}
    est2 = Ridge(alpha=0.25)
    #est2 = GridSearchCV(Ridge(), param_grid=param_grid_r, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')

    print("Fitting Classification models")
    ##Classification
    ##Fit the training data
    start1 = time()
    est1.fit(train, ytrain_c)
    ##Predict on the training data
    ypred_train_c = est1.predict(train)
    ##Predict on the test data
    ypred_test_c = est1.predict(test)
    print("Time taken for training and predicting (classification) is {} s".format(time()-start1))
    print()
    print("Metrics: \n")
    ##Metrics for training
    print("Classification report for training \n")
    print(classification_report(ytrain_c, ypred_train_c))
    print("Classification report for test \n")
    print(classification_report(ytest_c, ypred_test_c))
    ##Get the dominant class probabilities on the test set
    yproba_test_c = est1.predict_proba(test)
    ##Take only the max probability
    ypscore_test_c = [max(i) for i in yproba_test_c]
    ##For the roc_auc_score, take only the positive class 
    ydscore_test_c = [i[1] for i in yproba_test_c]
    print("AUC Score = {}".format(roc_auc_score(ytest_c, ydscore_test_c)))
    print()

    print("Fitting regression model")

    ##Regression
    ##Fit the training data
    start2 = time()
    est2.fit(train, ytrain_r)
    ##Predict on the training data
    ypred_train_r = est2.predict(train)
    ##Metrics (R2_score)
    print("Training R2 is {}".format(r2_score(ytrain_r,  ypred_train_r)))

    ##Predict on the test data
    ypred_test_r = est2.predict(test)
    ##Metrics (R2_score)
    print("Test R2 is {}".format(r2_score(ytest_r,  ypred_test_r)))
    print("Time taken for training and predicting (regression) is {} s".format(time()-start2))

    ##Get the results
    res = pd.concat([fl_test, 
                     pd.DataFrame(ytest_c, columns=['arr_delay']), 
                     pd.DataFrame(ypred_test_c, columns=['predicted_arr_delay']),
                     pd.DataFrame(ypscore_test_c, columns=['delay_or_nodelay_prob']), 
                     pd.DataFrame(ytest_r, columns=['arr_delay(minutes)']),
                     pd.DataFrame(ypred_test_r, columns=['predicted_arr_delay(minutes)'])], axis=1)

    display(res.head(20))
    
    ##Save estimators as dill objects
    dill.dump(est1, open("est1.dill","wb"))
    dill.dump(est2, open("est2.dill","wb"))

def opti_predict(origin, dest, fl_date, n=3, conn=None):
    colsp = ['fl_date', 'origin', 'origin_city_name', 'op_unique_carrier', 
             'op_carrier_fl_num', 'carrier_name', 'crs_dep_time', 'dest', 'dest_city_name',
            'crs_arr_time', 'odp']
    ##Get all the pertinent flights 
    query1 = "with fdict as (select t1.fl_date, t2.origin as origin, t1.origin as connecting_dest," + \
    " t2.op_unique_carrier as carrier_org, t2.crs_dep_time as origin_dep_time, t2.crs_arr_time as connecting_arr_time," + \
    " t2.air_time as leg1_at, t2.origin || '-' || t1.origin as leg1, t1.origin as connecting_origin," + \
    " t1.dest as dest, t1.op_unique_carrier as carrier_dest, t1.crs_dep_time as connecting_dep_time, " + \
    "t1.crs_arr_time as dest_arr_time, t1.air_time as leg2_at, t1.origin || '-' || t1.dest as leg2 " + \
    "from (select fl_date, origin, origin_city_name, op_unique_carrier, crs_dep_time, dest, crs_arr_time, air_time from test" + \
    " where dest = " + "'" + dest +  "'" + " and fl_date = "+ "'" + fl_date + "'" + ") t1 " + \
    "JOIN " + \
    "(select fl_date, origin, origin_city_name, op_unique_carrier, crs_dep_time, dest, dest_city_name, crs_arr_time, air_time from test " + \
    "where origin = " + "'" + origin + "'" + " and fl_date = " + "'" + fl_date + "'" + ") t2 " + \
    "on t1.origin = t2.dest " + \
    "where t2.crs_arr_time < t1.crs_dep_time and t2.crs_dep_time < t1.crs_dep_time and length(cast(t2.crs_arr_time as text)) > 3 " + \
    " and (t1.crs_dep_time - t2.crs_arr_time) > 60) " + \
    "select f1.*, c.description as carrier_name " +  \
    "from " + \
    "(select * from test where origin || '-' || dest in (select leg1 from fdict) and fl_date = " + "'" + fl_date + "'" + \
    " and crs_dep_time in (select origin_dep_time from fdict) " + \
    "and crs_arr_time in (select connecting_arr_time from fdict) " + \
    "and op_unique_carrier in (select carrier_org from fdict)) f1 " + \
    "join (select * from carriers) c on c.code = f1.op_unique_carrier " + \
    "UNION " + \
    "select f2.*, c.description as carrier_name " + \
    "from " + \
    "(select * from test where origin || '-' || dest in (select leg2 from fdict) " + \
    "and crs_dep_time in (select connecting_dep_time from fdict) " + \
    "and crs_arr_time in (select dest_arr_time from fdict) " + \
    "and op_unique_carrier in (select carrier_dest from fdict) " + \
    "and fl_date = " + "'" + fl_date + "'" + ") f2 " + \
    "join " + \
    "(select * from carriers) c on c.code = f2.op_unique_carrier " + \
    "UNION " + \
    "select f3.*, c.description as carrier_name " + \
    "from " + \
    "(select * from test where origin = " + "'" + origin + "'" + \
    " and dest = " + "'" + dest + "'" + " and fl_date = " + "'" + fl_date + "'" + ") f3" +\
    " join" + \
    "(select * from carriers) c on c.code = f3.op_unique_carrier order by odp"
    
    ##This is the query that will get all the links between origin and destination into a table
    query2 = "select t1.fl_date, t2.origin as source, t1.origin as connecting_dest," + \
    " t2.op_unique_carrier as carrier_org, t2.crs_dep_time as origin_dep_time, t2.crs_arr_time as connecting_arr_time," + \
    " t2.air_time as leg1_at, t2.origin || '-' || t1.origin as leg1, t1.origin as connecting_origin," + \
    " t1.dest as destination, t1.op_unique_carrier as carrier_dest, t1.crs_dep_time as connecting_dep_time, " + \
    "t1.crs_arr_time as dest_arr_time, t1.air_time as leg2_at, t1.origin || '-' || t1.dest as leg2 " + \
    "from (select fl_date, origin, origin_city_name, op_unique_carrier, crs_dep_time, dest, crs_arr_time, air_time from test" + \
    " where dest = " + "'" + dest +  "'" + " and fl_date = "+ "'" + fl_date + "'" + ") t1 " + \
    "JOIN " + \
    "(select fl_date, origin, origin_city_name, op_unique_carrier, crs_dep_time, dest, dest_city_name, crs_arr_time, air_time from test " + \
    "where origin = " + "'" + origin + "'" + " and fl_date = " + "'" + fl_date + "'" + ") t2 " + \
    "on t1.origin = t2.dest " + \
    "where t2.crs_arr_time < t1.crs_dep_time and t2.crs_dep_time < t1.crs_dep_time and length(cast(t2.crs_arr_time as text)) > 3 " + \
    " and (t1.crs_dep_time - t2.crs_arr_time) > 60"

    ##Read the test data into a pandas dataframe containing all the flights.
    dftest = pd.read_sql(query1, conn)
    #display(dftest)
    #print()
    #print(len(dftest))
    ##Read the table that contains the links into another dataframe
    dfopt  = pd.read_sql(query2, conn)
    #display(dfopt)
    #print()
    #print(len(dfopt))
    ##Now generate the relevant features for the test data
    ##Shuffle the dataframe first and pick a small subset of the original dataframe
    dftest = dftest.sample(frac=1).reset_index(drop=True)
    ##I really do not want to drop any of the data. 
    ##I'd have make sure that those flight segments are present 
    ##when I join the predictions dataframe.
    dftest = dftest.dropna().reset_index().drop('index',axis=1)

    ##Read the pipe estimators
    pipes = []
    names = ['pipe_time.dill', 'pipe_cong.dill', 'pipe_cat.dill', 'pipe_num.dill']
    for name in names:
        pipes.append(dill.load(open(name, "rb")))
    ##Feed the test data to the feature generator to generate the feature matrix
    Xtestdf = feature_gen(dftest, colsp, pipe=pipes, train=False, cong=True, rneg=False)
    
    #display(Xtestdf)
    ##Check for NaN rows
    #if Xtestdf.isnull().any():
    Xtestdf = Xtestdf.dropna().reset_index().drop('index', axis=1)

    ##Get the reals for the classification and regression problems
    yreal_c  = Xtestdf['arv_delay'].values
    yreal_r  = Xtestdf['arr_delay'].values

    ##drop target columns
    Xtestdf = Xtestdf.drop(['arr_delay', 'arv_delay'], axis=1)

    ##Now get the flight data for train and test, To be used for the network graph optimization model.
    fl_test  = Xtestdf[colsp]
    
    ##Now get the feature matrix
    colsf = [c for c in list(Xtestdf.columns) if c not in colsp]
    #print(colsf)
    test = Xtestdf[colsf].to_numpy()
    #display(test.shape)
    
    ##Load the predictors
    est1 = dill.load(open("est1.dill", "rb"))
    est2 = dill.load(open("est2.dill", "rb"))
    
    ##Make predictions
    ##Classification
    ypred_real_c = est1.predict(test)
    
#     print("Classification report for test \n")
#    print(classification_report(yreal_c, ypred_real_c))
    ##Get the dominant class probabilities on the test set
    yproba_real_c = est1.predict_proba(test)
    ##Take only the max probability
    #ypscore_real_c = [max(i) for i in yproba_real_c]
    ##For the roc_auc_score, take only the positive class 
    ydscore_real_c = [i[1]*100. for i in yproba_real_c]
#    print("AUC Score = {}".format(roc_auc_score(yreal_c, ydscore_real_c)))
#    print()
    
    ##Regression
    ypred_real_r = est2.predict(test)
    #Metrics (R2_score)
#    print("Test R2 is {}".format(r2_score(yreal_r,  ypred_real_r)))
    
    ##Merge the predictions with the fl_test dataframe
    res = pd.concat([fl_test, 
                     pd.DataFrame(yreal_c, columns=['arr_delay']), 
                     pd.DataFrame(ypred_real_c, columns=['predicted_arr_delay']),
                     pd.DataFrame(ydscore_real_c, columns=['delay_prob']), 
                     pd.DataFrame(yreal_r, columns=['arr_delay(minutes)']),
                     pd.DataFrame(ypred_real_r, columns=['predicted_arr_delay(minutes)'])], axis=1)
    
    
    ##Run the optimization module here
    fn_df = run_optimizer(res, dfopt)
    cols1 = ['fl_date', 'source', 'connecting_dest', 'carrier_org', 'carrier_name_leg1', 'op_carrier_fl_num_leg1',
         'origin_dep_time', 'connecting_arr_time', 'leg1', 'estimated_time_leg1(minutes)', 'layover_connecting',
        'estimated_arr_time_leg1 (HH:MM)', 'delay_prob_leg1']
    cols2 = ['fl_date', 'connecting_origin', 'destination', 'carrier_dest', 'carrier_name_leg2', 'op_carrier_fl_num_leg2', 
         'connecting_dep_time', 'dest_arr_time', 'leg2', 'estimated_time_leg2(minutes)', 'layover_connecting',
        'estimated_arr_time_leg2 (HH:MM)', 'delay_prob_leg2']
    fn_df1 = fn_df[cols1].rename(columns={'fl_date': 'Flight Date', 'source':'Origin', 'connecting_dest': 'Destination', 
                                          'carrier_org': 'Airline Code', 'carrier_name_leg1': 'Airline',
                                         'op_carrier_fl_num_leg1': 'Flight Number',
                                         'origin_dep_time': 'Recommended Departure Time (HH:MM)',
                                         'connecting_arr_time': 'Scheduled Arrival Time (HH:MM)',
                                         'leg1': 'Leg', 'estimated_time_leg1(minutes)': 'Trip Time (minutes)',
                                         'layover_connecting': 'Layover Time (minutes)',
                                         'estimated_arr_time_leg1 (HH:MM)': 'Estimated Arrival Time (HH:MM)',
                                         'delay_prob_leg1': 'Delay Probability (%)'})
    fn_df1['segment'] = list(range(1,len(fn_df1.index)*2+1, 2))
    fn_df1['Choice'] = list(range(1, len(fn_df1.index)+1))
    fn_df2 = fn_df[cols2].rename(columns={'fl_date': 'Flight Date', 'connecting_origin':'Origin', 'destination': 'Destination',
                                         'carrier_dest': 'Airline Code', 'carrier_name_leg2': 'Airline',
                                          'op_carrier_fl_num_leg2': 'Flight Number',
                                         'connecting_dep_time': 'Recommended Departure Time (HH:MM)',
                                         'estimated_time_leg2(minutes)': 'Trip Time (minutes)',
                                         'layover_connecting': 'Layover Time (minutes)',
                                         'dest_arr_time': 'Scheduled Arrival Time (HH:MM)',
                                         'leg2': 'Leg', 'estimated_arr_time_leg2 (HH:MM)': 'Estimated Arrival Time (HH:MM)',
                                         'delay_prob_leg2': 'Delay Probability (%)'})
    fn_df2['Layover Time (minutes)'] = np.nan
    fn_df2['Layover Time (minutes)'] = fn_df2['Layover Time (minutes)'].replace(np.nan, '', regex=True)
    fn_df2['segment'] = list(range(2,len(fn_df1.index)*2+1, 2))
    fn_df2['Choice'] = list(range(1, len(fn_df2.index)+1))
    fn_df_f = pd.concat([fn_df1, fn_df2]).sort_values(by='segment')
    fn_df_f['Recommended Departure Time (HH:MM)'] = fn_df_f['Recommended Departure Time (HH:MM)'].apply(lambda x: time_str(x))
    fn_df_f['Scheduled Arrival Time (HH:MM)'] = fn_df_f['Scheduled Arrival Time (HH:MM)'].apply(lambda x: time_str(x))
    #fn_df_f['Choice'] = fn_df_f.index
    fn_df_f = fn_df_f.reset_index().drop(['index','segment'], axis=1)
    cols_disp = ['Choice', 'Flight Date', 'Leg', 'Recommended Departure Time (HH:MM)', 
                 'Estimated Arrival Time (HH:MM)', 'Layover Time (minutes)', 
                'Trip Time (minutes)', 'Delay Probability (%)']
    fn_df_f[['Layover Time (minutes)', 'Trip Time (minutes)', 'Delay Probability (%)']] = (fn_df_f[['Layover Time (minutes)', 'Trip Time (minutes)', 'Delay Probability (%)']].round(2))
    return fn_df_f[cols_disp].head(n*2)


##Column select transformer
class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, col_names):
        self.col_names = col_names  # We will need these in transform()
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    
    def transform(self, X):
        # Return an array with the same number of rows as X and one
        # column for each in self.col_names
        
        return X[self.col_names]

class TimeGenerator(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, cols):
        self.cols = cols
        self.new_cols = ['sch_dep_time', 'sch_arr_time']
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        "Accepts a dataframe as input and normalizes the departure and arrival times"
        for c in range(len(self.new_cols)):
            X[self.new_cols[c]] = list(map(time_norm, X[self.cols[c]].values))
        X = X.drop(self.cols, axis=1)
        return X

##The goal is to use the sklearn's Feature_Union class to merge different features into a single datframe that 
##can be converted to a matix which will be fed into the ML framework.
class CongestionTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Takes a dataframe X and 
        returns the congestion for each 
        airport, fl_date and time_blk. The dataframe
        X has the following columns
        (fl_date, op_unique_carrier, origin, dep_time_blk, dest, arr_time_blk)  
        """
        ##Departure congestion
        Xdc = (X.groupby(['origin', 'fl_date', 'dep_time_blk'])['op_unique_carrier'].count()
               .reset_index()
               .rename(columns={'op_unique_carrier': 'dep_congestion', 'origin': 'code', 'dep_time_blk': 'time_blk'}))
        Xac = (X.groupby(['dest', 'fl_date', 'arr_time_blk'])['op_unique_carrier'].count()
               .reset_index()
               .rename(columns={'op_unique_carrier': 'arr_congestion', 'dest': 'code', 'arr_time_blk': 'time_blk'}))
        ##Merge the two dataframes
        Xc = Xdc.merge(Xac, left_on=['code', 'fl_date', 'time_blk'], right_on=['code', 'fl_date', 'time_blk'], how='outer')
        ##Fill  gaps with 0 and get the total congestion. The gaps needed to be filled with 0 so that 
        ##I don't get a NaN value for the total congestion.
        Xc = Xc.fillna(0)
        Xc['congestion'] = Xc['dep_congestion'] + Xc['arr_congestion']
        Xc = Xc.drop(['arr_congestion', 'dep_congestion'], axis=1)
        
        ##<Here I need to add a check for missing values. Add a try-catch-exception handling here>
        return Xc




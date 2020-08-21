# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:47:54 2020

@author: aditya
Capstone project: Flight itinerary recommender based on delays

"""
##Modules to import
from flask import Flask, render_template, request, redirect
import pandas as pd
#import numpy as np
#from time import time
from IPython.display import display
#from sys import exit
#import seaborn as sns
#import matplotlib.pyplot as plt
from db import DBConnector
from dpm import *


##Instantiate the Flask module (what is this doing really ?)
app_delay = Flask(__name__)

app_delay.vars = {}


@app_delay.route('/', methods=['GET'])
def index():
    return render_template("test1.html")

@app_delay.route('/fly', methods=['POST'])
def fly():
    app_delay.vars['origin'] = request.form['org']
    app_delay.vars['dest']   = request.form['dest']
    app_delay.vars['date']   = request.form['date']
    app_delay.vars['time']   = request.form['time']
    app_delay.vars['topn']   = request.form['topn']
    
    print(app_delay.vars['origin'], app_delay.vars['dest'], app_delay.vars['date'], 
          app_delay.vars['time'], app_delay.vars['topn'])
    print(type(app_delay.vars['origin']), type(app_delay.vars['dest']), 
          type(app_delay.vars['date']), type(app_delay.vars['time']),
          type(app_delay.vars['topn']))
    
    
    ##Connect to the db
    db = r"C:\Users\adity\Documents\Data_Science\Data_Incubator\data\sqlite\aviation_delay.db"
    dbc = DBConnector()
    conn = dbc.create_connection(db)
    print(conn)
    
    print("Getting recommendations")
    fn_df = opti_predict(app_delay.vars['origin'], 
                         app_delay.vars['dest'], app_delay.vars['date'],
                         n=int(app_delay.vars['topn']), 
                         conn=conn)
    #display(fn_df)
    return render_template('fly.html', tables=[fn_df.to_html(classes='flight', 
                                                             header="true", 
                                                             justify='center',
                                                             index=False)])
        
    # p = displayStock(app.vars['symbol'], app.vars['params'])
    # try:
    #     script, div = components(p)
    #     return render_template("plot.html", script=script, div=div)
    # except ValueError as e:
    #     return ("That ticker symbol %s is invalid! Try entering another symbol."%app.vars['symbol'])
# def main():
    
#     db = r"C:\Users\adity\Documents\Data_Science\Data_Incubator\data\sqlite\aviation_delay.db"
#     dbc = DBConnector()
#     conn = dbc.create_connection(db)
#     print(conn)
    
#     origin = 'SFO'
#     dest = 'JAX'
#     fl_date = '2019-03-01'
#     ##Predict using the opti_predict
#     print("Getting recommendations")
#     fn_df = opti_predict(origin, dest, fl_date, conn=conn)
#     display(fn_df)

if __name__ == "__main__":
    app_delay.run(port=5000, debug=True)
    #main()
    
    
    
    






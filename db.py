# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 10:43:48 2020

@author: aditya
"""

import sqlite3
##Database Connector class
class DBConnector:
    
    ##Create a SQLite DB for the Data Incubator project called 'aviation_delays.db'
    def create_connection(self, db_file):
        """ create a database connection to a SQLite database 
        Input
        :param db_file: file name of the db
        Output
        :conn connection to the sqlite db server
        """
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            print(sqlite3.version)
            return conn
        except sqlite3.Error as e:
            print(e)

        return conn
    
    ##Let's use Python to create a function that creates/appends to a table in a db.
    ##This needs to be a generic function that will take a table name and relevant 
    ##data in the form of a csv file first read the data into a pandas dataframe and 
    ##then extract the columns from the dataframe with their relevant types and then put 
    ##put the data into a database table.
    def table_generator(self, df, conn, table_name):
        """Create a table if it does not exist or append to an 
        existing table in a database.
        Inputs:
        df: Pandas dataframe containing the data.
        conn: Connection to the database
        table_name: name of the table to create or append.
        Output:
        flag: Flag that indicates whether the table was successfully created or not."""

        ##Creating table and insert values into it from the data frame.
        print('Writing to table '+table_name)
        #start = time()
        df.to_sql(table_name, con=conn, if_exists='append', index=False)
        #print('Total write time = ',time()-start)
        ##To see if the table has been successfully created
        try:
            c = conn.cursor()
            c.execute("SELECT * FROM " + table_name)
            print(c.fetchone())
            success = True
        except sqlite3.Error as e:
            print(e)
            sucess = False

        return success

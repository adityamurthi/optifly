# optifly
This repository contains code that predicts the most optimal time to fly based on predicting airline delays. It uses data from multiple sources such as flight data from the Bureau of Transportation Statistics (BTS) (https://www.transtats.bts.gov/Tables.asp?DB_ID=120&DB_Name=Airline%20On-Time%20Performance%20Data&DB_Short_Name=On-Time), weather data from NWS (https://mesonet.agron.iastate.edu/request/download.phtml) and aircraft data from the FAA (https://www.faa.gov/licenses_certificates/aircraft_certification/aircraft_registry/releasable_aircraft_download/). The code uses data from these threee different sources to first predict whether a individual flights will be delayed, using both a classifier (delay = 1, no delay = 0) and a regressor (predicts delay in minutes). It then takes the predicted delay minutes for each flight in a flight plan and uses Dikikstra's shortest path algorithm to find the "best" flying time based on the route that minimizes the total trip time which incorporates the delay times. 

The repository consists of the following pieces of code, pickled estimator objects, directories containing HTML code for the interactive webpage as well as requirements.txt file to install all the packages.  
1. app_delay.py -- Implements an interactive OptiFly webpage using Flask, that allows a user to specify origin, destination, travel date a time preference and whether he/she wants to optimize on the total trip time or cost. 
2. dpm.py -- Contains the main code to predict delays as well as the optimizer that picks the best flight time and route. 
3. db.py -- Contains code to interact with the Sqlite3 database where all the data is stored. 

To run the app follow these instructions:
* `$ python app_delay.py` -- to start the Flask app that provides a link to the interactive webpage. This is currently set to `127.0.0.1:5000/` on you local machine > NOTE: I use a Windows 10 PC so this link might be different for different operating sytems.
* Navigate to this page and enter the source, destination, flight date, travel time preference (if any), optimize on Trip time (**only this implemented for now**) and number of flight recommendations to return. Click `GoFly` and obtain predictions.


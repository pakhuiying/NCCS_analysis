import requests
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import importlib
import LTA_API_key
importlib.reload(LTA_API_key)
import LTA_API_key as apiKeys
from datetime import datetime

def generate_OneMap_token():
    """ generates new one map token """
    onemapKey = apiKeys.get_OneMap_token()
    headers = {"Authorization": onemapKey}
    return headers

def get_OneMap_response(url,headers):
    """ 
    Args:
        url (str): url for request
        headers (dict): Authorization headers
    """
    response = requests.request("GET", url, headers=headers)
    response = response.json()
    return response

def get_OneMap_itineraries(headers,start_lat,start_lon,end_lat,end_lon,date,time ,
                           route_type = "pt", mode = 'TRANSIT', maxWalkDistance = 1000, numItineraries = 3):
    """
    Args:
        headers (dict): Authorization headers
        start_lat (float): latitude coordinate of where you start your journey
        start_lon (float): longitude coordinate of where you start your journey
        end_lat (float): latitude coordinate of where you end your journey
        end_lon (float): longitude coordinate of where you end your journey
        route_type (str): "pt" # Route types available walk, drive, pt, and cycle. Only lowercase is allowed.
        date (str):  e.g. '01-13-2025' Date of the selected start point in MM-DD-YYYY.
        time (str): e.g. '07%3A35%3A00' Time of the selected start point in [HH][MM][SS], using the 24-hour clock system. 
        mode (str): e.g. 'TRANSIT'.  Mode of public transport: TRANSIT, BUS, RAIL. Entry must be specified in UPPERCASE
        maxWalkDistance (float): e.g. 1000. The maximum walking distance set by the user in metres.
        numItineraries (int): maximum number if possible itineraries to fetch
    """
    url = f"https://www.onemap.gov.sg/api/public/routingsvc/route?start={start_lat}%2C{start_lon}&end={end_lat}%2C{end_lon}&routeType={route_type}&date={date}&time={time}&mode={mode}&maxWalkDistance={maxWalkDistance}&numItineraries={numItineraries}"

    response = get_OneMap_response(url, headers)
    try:
        itineraries = response['plan']['itineraries']
    except Exception as e:
        print(f'Token may have expired, input new headers: {e}')
    return itineraries


class OneMapItinerary:
    def __init__(self, itinerary):
        """ 
        Args:
            itinerary (dict): an itinerary, where keys are the metadata of the itinerary. Itineraries are obtained from OneMap's routing API
            GTFS (pd.DataFrame): a dataframe created from joining GTFS's stop_times and stops, such that stop sequence, and its stop ID and coordinates are captured
        """
        self.itinerary = itinerary
        self.duration = itinerary['duration'] # in seconds
        self.startTime = itinerary['startTime']
        self.endTime = itinerary['endTime']
        self.transitTime = itinerary['transitTime']
        self.waitingTime = itinerary['waitingTime']
        self.transfers = itinerary['transfers']
        self.n_legs = len(itinerary['legs'])
        self.legs = itinerary['legs']
        # TODO: only filter legs that have mode == 'BUS'
    
    def get_bus_legs(self):
        """ 
        filter legs where mode == 'BUS'
        """
        return [l for l in self.legs if l['mode']=='BUS']
    
    def get_bus_routes(self):
        """ 
        concatenate all bus legs to get the intermediate bus stops
        """
        bus_legs = self.get_bus_legs()
        busLegs = [BusLeg(b) for b in bus_legs]
        # add to attribute
        self.busLegs = busLegs
        busLegs_dfs = [B.get_stops_data() for B in busLegs]
        busLeg_df = pd.concat(busLegs_dfs)
        return busLeg_df

    def get_total_distance(self):
        """ 
        must have already ran `get_bus_routes`
        calculate total distance for all the bus legs
        """
        return sum([B.distance for B in self.busLegs])
    
    def get_total_duration(self):
        """ 
        must have already ran `get_bus_routes`
        calculate total duration for all the bus legs
        """
        return sum([B.duration for B in self.busLegs])

class BusLeg:
    def __init__(self, busLeg):
        """ 
        Args:
            busLeg (dict): a bus leg, where keys are meta data for the bus leg
            GTFS (pd.DataFrame): dataframe of the GTFS's shapes file
        """
        self.busLeg = busLeg
        self.duration = busLeg['duration'] # in seconds
        self.distance = busLeg['distance'] # in metres
        self.startTime = busLeg['startTime']
        self.endTime = busLeg['endTime']
        
        self.mode = busLeg['mode']
        self.routeId = busLeg['routeId'] # bus number

    def get_stops_data(self):
        """ 
        Args:
            stops (dict): with keys: 'arrival','departure','lat','lon','name','stopCode','stopId','stopIndex','stopSequence','vertexType'
        Returns:
            pd.DataFrame: shows a sequence of bus stops at 'from' bus stop to 'to' bus stop
        """
        stops_list = [self.busLeg['from']] + self.busLeg['intermediateStops'] + [self.busLeg['to']]
        stops_df = pd.DataFrame(stops_list)
        
        return stops_df


        
        
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 20:58:03 2019

@author: hydro
"""

import requests

response = requests.get('https://api.github.com/events')

response_json = response.json()
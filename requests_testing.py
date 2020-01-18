# -*- coding: utf-8 -*-
#https://requests.kennethreitz.org/en/master/user/quickstart/
"""
Created on Thu Nov  7 20:58:03 2019

@author: hydro
"""

import requests

#get
'''response = requests.get('https://api.github.com/events')
response_json = response.json()'''

#post
post_req = requests.post('https://httpbin.org/post', data={'form':{
    "comments": "Coooooooked fast", 
    "custemail": "h@1", 
    "custname": "Karan", 
    "custtel": "123456", 
    "delivery": "11:45", 
    "size": "small", 
    "topping": [
      "cheese", 
      "onion", 
      "mushroom"
    ]
  }})


#READING GOOGLE BOOKS, didnt see where the API key was used, no use in the following requsts
f = open(r'c:\Users\Karan.Verma\Downloads\gbooksapikey\api key.txt')
contents = f.read()

r = requests.get('GET https://www.googleapis.com/books/v1/volumes?q=quilting')

json_response = r.json()
json_out = open(r'C:\Users\Karan.Verma\Downloads\gbooksapikey\json_load.json', mode='w')
for values in json_response['items']:
	json_out.write(values['volumeInfo']['title'])
	json_out.write(str(values['volumeInfo']['authors']))
	json_out.write(values['volumeInfo']['publisher'])
	json_out.write(values['volumeInfo']['description'])
	json_out.write('\n')

json_out.close()




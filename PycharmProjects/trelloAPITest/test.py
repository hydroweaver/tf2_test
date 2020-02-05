import requests
import json

api_token = 'API_TOKEN'
api_key = 'API_KEY'
trello_url = str('https://api.trello.com/1/members/me/boards?key=' + api_key + '&token=' + api_token)

trello_info = requests.get(trello_url)

if trello_info.status_code == 200:
    trello_response_json = trello_info.json()
    for boards in trello_response_json:
        if boards['name'] == 'Delhi Trip':
            print('Found Board named "Delhi Trip"')
            print(trello_response_json[0])
            break
        else:
            print('Board with name "Delhi Trip" not found!')

else:
    print(trello_info.status_code)



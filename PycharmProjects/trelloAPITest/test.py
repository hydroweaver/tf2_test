import requests
import json

api_token = 'a15fc0880a68ab072cb2b5132832703d46a46107257c32c9c5a355e8012120cb'
api_key = '835888a2e5125840051c10fd80c58606'
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



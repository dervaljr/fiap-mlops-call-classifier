import json
import src.app as app

with open('data.json') as data_file:
    event = json.load(data_file)

app.handler(event, False)
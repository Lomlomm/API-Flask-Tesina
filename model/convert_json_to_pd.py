import pandas as pd
import requests

def Convert2DF():
    url = 'http://localhost:5000/processData'
    response = requests.get(url)
    data = response.json()
    response_data = data['response']

    df = pd.json_normalize(response_data)
    return df


#%%
import json
import os
import time
import uuid
from datetime import datetime, timedelta
import boto3
from collections import defaultdict
import numpy as np
import random
import pandas as pd
import ast
#%%
#table_names

SPANISHUSERS = 'dev-languageApp-spanishUsers'
NOTIFICATIONS = 'dev-languageApp-Notifications'
USERACTIONS = 'dev-languageApp-ChatterBoxuserActions'
VOCABULARY = 'dev-languageApp-ChatterBoxVocab'
#%%
ACCESS_KEY = ''
SECRET_ACCESS_KEY = ''
#%%
# 1) Initialize AWS Clients and Configuration

dynamodb_client = boto3.client(
    'dynamodb',
    region_name='us-east-1',  # e.g., 'us-west-2'
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_ACCESS_KEY
)

athena_client = boto3.client("athena",
                             aws_access_key_id=ACCESS_KEY,
                             aws_secret_access_key=SECRET_ACCESS_KEY,
                             region_name="us-east-1")
#%%
# 2) DynamoDB Scan for Users

#Define scan parameters
params = {
    "TableName": SPANISHUSERS
}

try:
    response = dynamodb_client.scan(**params)
    items = response.get("Items", [])
    print("Retrieved Items:", items)
except Exception as e:
    print("Error fetching data:", e)
#%%
# Extract valid (email, token) pairs
users_info = []
for item in items:
    email = item.get("Email", {}).get("S", "")
    token = item.get("FCM_Token", {}).get("S", "")
    streak_status = item.get("Streak", {}).get("N", "")
    streak_change = item.get("Last_Streak_Change", {}).get("S", "")
    daily_availability = item.get("Daily_Availability", {}).get("S", "")
    #if email and token:
    if email:
        users_info.append((email, token, streak_status, streak_change, daily_availability))
#%%
# 4) GET VOCABULARY WORDS FROM DYNAMODB
params = {
    "TableName": VOCABULARY
}

all_words = []

try:
    while True:
        response = dynamodb_client.scan(**params)
        all_words.extend(response.get("Items", []))

        # Check if there are more items to retrieve
        if 'LastEvaluatedKey' in response:
            params['ExclusiveStartKey'] = response['LastEvaluatedKey']
        else:
            break

    print(f"Retrieved {len(all_words)} Rows from dev-languageApp-ChatterBoxVocab")

except Exception as e:
    print("Error fetching data:", e)

words = []

for item in all_words:
    # Flatten each dictionary by extracting the value from the nested structure
    converted_item = {key: value['S'] for key, value in item.items()}
    words.append(converted_item)
    
#%%
words
#%%
unique_words = pd.DataFrame(words)[['Targ_Word','Base_Word','Level','Base_Lang_Code']]
unique_words = unique_words[unique_words['Base_Lang_Code'] == 'EN']
unique_words = unique_words[['Targ_Word','Base_Word','Level']]

unique_words = unique_words.drop_duplicates().sort_values(by=['Level','Targ_Word'])
unique_words = unique_words[unique_words['Level'] != '1']
unique_words
#%%
unique_words.to_csv('unique_words.csv',index=False)
#%%

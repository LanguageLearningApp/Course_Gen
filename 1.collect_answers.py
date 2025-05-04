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
#%%
#table_names

SPANISHUSERS = 'dev-languageApp-spanishUsers'
NOTIFICATIONS = 'dev-languageApp-Notifications'
USERACTIONS = 'dev-languageApp-ChatterBoxuserActions'
#%%
ACCESS_KEY = ''
SECRET_ACCESS_KEY = ''


athena_database = "chatterbox-analytics-dev"
athena_table = "dev_languageapp_useractions_vw"
athena_output_location = "s3://grantj3-languageapp-langu-chatterboxanalyticsbucke-9ifdazfz8y6p/athena-results/"
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
# 2) DynamoDB Scan for FCM token

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
chunk_size = 100

user_emails = [u[0] for u in users_info]
email_list_str = ", ".join(f"'{email}'" for email in user_emails)
chunks = [user_emails[i:i + chunk_size] for i in range(0, len(user_emails), chunk_size)]

all_results = []

for chunk in chunks:
    # Generate user_id placeholders for the filter expression
    email_placeholders = ', '.join([f":user_id{i}" for i in range(len(chunk))])

    # Initialize the scan parameters
    params = {
        "TableName": USERACTIONS,
        "FilterExpression": f"user_id IN ({email_placeholders}) AND #timestamp <> :empty",
        "ExpressionAttributeNames": {
            "#timestamp": "timestamp"  # Escaping the reserved keyword
        },
        "ExpressionAttributeValues": {
            # Dynamically add user_id values as placeholders
            **{f":user_id{i}": {"S": chunk[i]} for i in range(len(chunk))},
            ":empty": {"S": ""}
        },
    }

    # Handle pagination
    while True:
        # Perform the scan
        response = dynamodb_client.scan(**params)

        # Extend the results
        all_results.extend(response.get("Items", []))

        # Check if there are more results (pagination)
        if "LastEvaluatedKey" in response:
            params["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        else:
            break

# Print the final count of items
print(f"Retrieved {len(all_results)} items.")

athena_events = []

for item in all_results:
    # Flatten each dictionary by extracting the value from the nested structure
    converted_item = {key: value['S'] for key, value in item.items()}
    athena_events.append(converted_item)

for event in athena_events:
    event['event_timestamp'] = event.pop('timestamp')

df_sorted = pd.DataFrame(athena_events).sort_values(by=['event_timestamp', 'user_id'], ascending=[True, True])
#%%
df_sorted
#%%
df_sorted.groupby('event_type').count()
#%%
df = df_sorted[df_sorted['event_type'].isin(['Push_Notification_Sent'])][['user_id','event_type','event_detail','section','event_id','event_timestamp']]
df
#%%
answers = df_sorted[df_sorted['event_type'].str.contains('orrect', na=False)]
answers
#%%
answers = df_sorted[df_sorted['event_type'].str.contains('orrect', na=False)]
answers['event_type'] = answers['event_type'].str.lower()

answers['result'] = answers['event_type'].str.extract(r'_(correct|incorrect)$')[0]

# Extract type by removing the result suffix
answers['type'] = answers['event_type'].str.replace(r'_(correct|incorrect)$', '', regex=True)

answers = answers[['event','user_id','event_timestamp','section','section_level','type','event_detail_2','event_detail_3','result']]

answers = answers[answers['type'].isin(['accent','mark','sentence'])]

answers['result_binary'] = answers['result'].map({'correct': 1, 'incorrect': 0})

answers
#%%
answers['event_detail_2'] = answers['event_detail_2'].str.replace(
    r'^.*Correct Spanish Word:\s*', '', regex=True
)

answers['event_detail_2'] = answers['event_detail_2'].str.replace(
    r'^.*Correct Spanish:\s*', '', regex=True
)



answers
#%%
answers.to_csv('Answers_Formatted.csv',index=False)
#%%

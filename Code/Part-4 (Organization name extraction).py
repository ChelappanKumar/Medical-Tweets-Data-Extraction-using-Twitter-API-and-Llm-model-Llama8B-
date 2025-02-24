import pandas as pd
import json
from langchain_ollama import ChatOllama
from tqdm import tqdm
import os
import re
import re


tqdm.pandas()
input_file = r"Output_Part-3 (Role classification)\tweets_role_classification.csv" #Specify the path for input file
output_file = r"Output_Part-4 (Org name extraction)\classified_organizations1.csv" #Specify the path for output file


df = pd.read_csv(input_file,dtype={'original_tweet_id':str,'author_id':str,"tweet_id":str})
model = ChatOllama(model="llama3.1:8b", temperature=0)
def extract_organization_from_bio(user_bio):
    if pd.isna(user_bio):
        return None
    try:
        # prompt = (
        #     f"""Extract the name of the organization where the user is currently working from the following bio: '{user_bio}'.\n
        #     If the bio is empty or no organization is mentioned, return None.\n
        #     Provide only the result."""
        # )
        prompt = f"""
Identify the organization where the user is currently working based on the following bio: 

'{user_bio}'

- If an organization name is present, return only the name of the organization.
- If the bio does not mention an organization, return "None".
- Do NOT return any additional text or explanation.
"""

        response = model.invoke(prompt)
        model_response = response.content.strip()
        return model_response
    except Exception as e:
        print(f"Error extracting organization: {e}")
        return None
df['organization'] = df['description'].progress_apply(extract_organization_from_bio)
df['organization'] = (df['organization'].replace("none", "").replace("None", ""))
df['tweet_id'] = df['tweet_id'].astype(str)
df['author_id'] = df['author_id'].astype(str)
df['original_tweet_id'] = df['original_tweet_id'].astype(str)
df['original_tweet_id']=df['original_tweet_id'].replace("nan","").replace("NaN","")
df['organization'] = df['organization'].str.replace(r'''[@#"\']''', '', regex=True)
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Results saved to {output_file}")
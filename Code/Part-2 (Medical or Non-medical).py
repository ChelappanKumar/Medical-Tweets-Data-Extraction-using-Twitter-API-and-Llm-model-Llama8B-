import pandas as pd
from langchain_community.chat_models import ChatOllama
import json
from tqdm import tqdm

tqdm.pandas()

# Load the CSV file
input_file = r"Output_Part-1 (scraping tweets)\test.csv" #Specify the path for input file
output_file = r"Output_Part-2 (Medical or Non-medical)\classified_tweets.csv" #Specify the path for output file

df = pd.read_csv(input_file,dtype={'original_tweet_id':str,'author_id':str,"tweet_id":str})
# df = df.head()
print(df.columns)
model = ChatOllama(model="llama3.1:8b", temperature=0)

classification_prompt = """
You are a classification model tasked with determining whether a given tweet is related to medical topics or not. Your goal is to classify the tweet based on its content.

There are two categories:  
1. **Medical**: The tweet includes information related to health, diseases, treatments, medical procedures, medications, healthcare, or any specific topic directly connected to the medical field.  
2. **Non-Medical**: The tweet contains medical terms or words but does not communicate actual medical information. This category also includes general tweets about lifestyle, entertainment, politics, social issues, and other non-medical subjects.

Message: {message}

### How to Classify:
- **Medical**: Classify as "Medical" if the tweet is directly or indirectly related to medical knowledge, health discussions, or specific health issues (e.g., “cancer treatment,” “vaccines,” “mental health awareness,” “heart surgery”).
- **Non-Medical**: Classify as "Non-Medical" if the tweet discusses health-related terms but lacks any context about medical treatment, diagnosis, or healthcare (e.g., “exercise is good for your health,” “I’m feeling sick,” “taking vitamins”).

### Instructions:
1. Carefully read the tweet and analyze its context to determine its category.  
2. Provide your classification result **ONLY** in the following JSON format:  

Provide the result strictly in the following JSON format:
{{ "category": "Medical/Non-Medical", "confidence": <confidence_score> }}

**Important**:  
- Always return the result **only** in JSON format, without any additional words, comments, or explanations.  
- Any response, regardless of input or context, should strictly adhere to this JSON format. Do not include any extra content or text outside the JSON.
- Do not include any extra content, words, or letters outside the JSON. Ensure the output contains **only** the JSON format specified above.

"""


# Define the classify_tweet function
model_response_list = []
def classify_tweet(row):
    tweet = row['text']
    # user_bio = row['User_Bio.1']
    # print("user_bio : ",user_bio) 
    try:
        # Format the prompt
        prompt = classification_prompt.format(message=tweet)
        
        # Invoke the model
        response = model.invoke(prompt)
        model_response = response.content.strip()
        model_response_list.append(model_response)
        
        print(f"Model response: {model_response}")

        # Parse the JSON output
        try:
            classification = json.loads(model_response)
            if all(key in classification for key in ['category', 'confidence']):
                return classification
            else:
                return {"category": "Error", "confidence": 0.0}
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {model_response}")
            return {"category": "Error", "confidence": 0.0}

    except Exception as e:
        print(f"Error classifying tweet: {e}")
        return {"category": "Error", "confidence": 0.0}

# Apply the classification function
df['classification'] = df.progress_apply(classify_tweet, axis=1)

# Extract JSON fields into separate columns
df['category'] = df['classification'].apply(lambda x: x.get('category', 'Error'))
df['confidence'] = df['classification'].apply(lambda x: x.get('confidence', 0.0))
# df['type_of_user'] = df['classification'].apply(lambda x: x.get('type_of_user', 'Unknown'))

# Store raw JSON responses for debugging or analysis
df['raw_json_classification'] = model_response_list
df['tweet_id'] = df['tweet_id'].astype(str)
df['author_id'] = df['author_id'].astype(str)
df['original_tweet_id'] = df['original_tweet_id'].astype(str)
df = df.drop(columns=['classification', 'confidence','raw_json_classification'])
df['original_tweet_id']=df['original_tweet_id'].replace("nan","").replace("NaN","")

# Save the final DataFrame to a CSV file
df.to_csv(output_file, index=False , encoding= 'utf-8-sig')
print(f"Classifications saved to {output_file}")


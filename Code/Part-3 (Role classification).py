import re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
from tqdm import tqdm  # Ensure tqdm is imported for progress_apply to work

# Initialize tqdm with pandas
tqdm.pandas()

# Load data
input_file = r"Output_Part-2 (Medical or Non-medical)\classified_tweets.csv" #Specify the path for input file
output_file = r"Output_part-3 (Role classification)\tweets_role_classification.csv" #Specify the path for output file

df = pd.read_csv(input_file,dtype={'original_tweet_id':str,'author_id':str,"tweet_id":str})

# Define the LLM model
llm = OllamaLLM(model="llama3.1:8b", temperature=0)

# Define the classification prompt template
classification_prompt_template = """
You are an advanced language model designed to classify the role of Twitter users discussing multiple sclerosis (MS) and its medicinal drugs. Your task is to:

1. Determine the User's Role based on their tweet and bio as either "patient", "doctor", "organisation" or "others" according to the rules below.
2. Generate a Description summarizing the user's perspective on MS-related topics in their tweet.

Classification Rules:
- Patient:
    - The tweet expresses personal experience, e.g., "I used", "I tried", "My doctor prescribed".
    - The user could also be a doctor experiencing MS themselves (even if their bio states they are a doctor).
- Doctor:
    - The user bio explicitly mentions being a doctor, neurologist, or MS specialist.
    - The user's tweets should reflect professional insights, medical advice, or observations from a doctor's perspective.
- Others:
    - If the user participating in awarness or fundrising event.
    - If the user shares the experenice of other person, also contains words like comments on.
    - If user bio is doesnt contain any value,none,null,empty.
- Organisation
    - The user bio indicates that if it is an organisation.


Task:
Given the tweet and user bio, classify the user into one of the roles (patient, doctor, or others) and provide a brief description summarizing the tweet's perspective.

Return Format:
"User's Role"="<role>"
"Description"="<summary>"
Note:
-there should be no text other than the return format strictly even the double quotes
tweet:
{tweet}
user_bio:
{bio}
"""

# Initialize prompt
classification_prompt = PromptTemplate(input_variables=["text", "description"], template=classification_prompt_template)

# Create the LLM chain
chain = LLMChain(llm=llm, prompt=classification_prompt)

# Function to classify user role and summarize perspective
def classify_user(row):
    tweet = row['text']
    bio = row.get('description', '')  # Use an empty string if 'bio' column is missing
    try:
        # Run the chain
        response = chain.run({"tweet": tweet, "bio": bio})
        
        # Print raw response for debugging
        print(f"Raw response: {response}")

        # Parse the response
        user_role = re.search(r'"User\'s Role"="(.+?)"', response)
        print('user_role',user_role)
        description = re.search(r'"Description"="(.+?)"', response)

        return pd.Series({
            "User's Role": user_role.group(1) if user_role else "Error",
            "Description": description.group(1) if description else "Error"
        })

    except Exception as e:
        print(f"Error for tweet '{tweet}': {e}")
        return pd.Series({
            "User's Role": "Error",
            "Description": str(e)
        })

# Apply the function to the DataFrame and add the results as new columns
df_classification = df.progress_apply(classify_user, axis=1)

# Concatenate the original DataFrame with the classification results
df = pd.concat([df, df_classification], axis=1)
df['tweet_id'] = df['tweet_id'].astype(str)
df['author_id'] = df['author_id'].astype(str)
df['original_tweet_id'] = df['original_tweet_id'].astype(str)
df['original_tweet_id']=df['original_tweet_id'].replace("nan","").replace("NaN","")
df = df.rename(columns={'Description': 'Insight'})
# Save the updated DataFrame to an Excel file
df.to_csv(output_file, index=False,encoding='utf-8-sig')
print(f"Saved to {output_file}")
import openai
import pandas as pd
import pickle

from config import OPENAI_API_KEY, COMPLETIONS_MODEL

from openai_functions import construct_prompt

openai.organization = "org-h2tLuOD0WsmSH4extTGzgOXU" #this is identical to other organization keys. this is fine to share
openai.api_key = OPENAI_API_KEY
openai.Model.list()

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": .2,
    "max_tokens": 500,
    "model": COMPLETIONS_MODEL,
}

#LOAD THE EMBEDDINGS FROM A PKL FILE
path_pkl = "data/first_1000_embeddings.pkl"
with open(path_pkl, 'rb') as f:
    doc_embeddings = pickle.load(f)

# NEED TO LOAD the dataset so that the paragraphs can be returned with the dictionary
df = pd.read_csv('data/FULL_DATA_short.csv')
df = df.rename(columns={'Unnamed: 0': 'index'})
small_df = df.iloc[:1000] #this will just have the first X rows
small_df = small_df.set_index(["title", "section", "subsection", "p_number", "index"])

# START THE Q&A PROCESS
while True:
    # question = "Please compare abduction to Bayesian Confirmation Theory"
    question = input("Ask a philosophy related question! ")

    if question == "Exit" or question == "exit":
        break
    prompt_sample, _ = construct_prompt(question, doc_embeddings, small_df)

    response = openai.Completion.create(
                    prompt=prompt_sample,
                    **COMPLETIONS_API_PARAMS
                )

    print(response["choices"][0]["text"].strip(" \n"))

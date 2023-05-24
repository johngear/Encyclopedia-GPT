import openai
import pandas as pd
import pickle
import time

import _pickle as cPickle

from config import OPENAI_API_KEY, COMPLETIONS_MODEL

from openai_functions import construct_prompt
from UPDATED_openai_functions import UPDATE_construct_prompt

openai.organization = "org-h2tLuOD0WsmSH4extTGzgOXU" #this is identical to other organization keys. this is fine to share
openai.api_key = OPENAI_API_KEY
openai.Model.list()

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": .2,
    "max_tokens": 500,
    "model": COMPLETIONS_MODEL,
}

##STARTING THE TIMER BEFORE WE LOAD DATASET
start = time.time()

##### New Data Format ######
with open('data/full_data/numpy/embeddings_full.pickle', 'rb') as file:
    doc_embeddings2 = pickle.load(file)


# NEED TO LOAD the dataset so that the paragraphs can be returned with the dictionary
df = pd.read_csv('data/FULL_DATA_new.csv')
df = df.rename(columns={'Unnamed: 0': 'index'})


# small_df = df.set_index(["title", "section", "subsection", "p_number", "index"])
# small_df = df.iloc[:1000] #this will just have the first X rows
# small_df = small_df.set_index(["title", "section", "subsection", "p_number", "index"])

print(f'Time for Loading Dataset into DF: {time.time() - start}')


# TRYING NEW INPUT FROM HERE

# START THE Q&A PROCESS
while True:
    # question = "Please compare abduction to Bayesian Confirmation Theory"
    question = input("Ask a philosophy related question! ")
    

    if question == "Exit" or question == "exit":
        break

    start = time.time()
    # prompt_sample, _ = construct_prompt(question, doc_embeddings2, df)
    prompt_sample, _ = UPDATE_construct_prompt(question, doc_embeddings2, df)
    print(f'Time for Constructing Prompt: {time.time() - start}')

    response = openai.Completion.create(
                    prompt=prompt_sample,
                    **COMPLETIONS_API_PARAMS
                )

    print(response["choices"][0]["text"].strip(" \n"))
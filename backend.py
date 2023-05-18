import openai
import pandas as pd
import pickle
import time

import _pickle as cPickle

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

##STARTING THE TIMER BEFORE WE LOAD DATASET
start = time.time()

##### PARTIAL DATASET ######


path_pkl = "data/first_1000_embeddings.pkl"
with open(path_pkl, 'rb') as f:
    doc_embeddings2 = pickle.load(f)
"""
"""
# df_full = pd.read_csv('data/full_data/FULL_INFO_PARALLEL.csv')
# doc_embeddings2 = df_full.to_dict(orient='records')
# print(f'Time for Loading Embeddings: {time.time() - start}')
# start = time.time()


##### PARTIAL END #####

##### RECALCULATE FULL THING #####

from read_jsonl import read_json

"""
# filename = 'data/full_data/full_info_1000_parallel.jsonl'
filename = 'data/full_data/FULL_INFO_PARALLEL.jsonl'
#TODO need to speed this part up
doc_embeddings2 = read_json(filename)
"""

##### RECALCULATE END #####

##### FULL DATASET #####
"""
with open('data/full_data/FULL_INFO_PARALLEL_last.pickle', 'rb') as handle:
    doc_embeddings2 = cPickle.load(handle)
    # doc_embeddings2 = pd.read_pickle(handle)
handle.close()

##GETTING RID OF PART OF THE DICTIONARY!
import random
desired_removal = int(len(doc_embeddings2) * 0.95)
items = list(doc_embeddings2.items())
random.shuffle(items)

del doc_embeddings2

items_to_remove = items[:desired_removal]
remaining_items = items[desired_removal:]
doc_embeddings2 = dict(remaining_items)
"""
##### FULL DATASET DONE #####

#format:
# tuple('title', 'section', 'subsection', 'paragraph number', 'index')

# doc_embeddings_2 = pd.read_csv('data/MERGED_small_fast_embedding.csv')
# dict2 = doc_embeddings_2.to_dict('embedding')



# NEED TO LOAD the dataset so that the paragraphs can be returned with the dictionary
df = pd.read_csv('data/FULL_DATA_new.csv')
df = df.rename(columns={'Unnamed: 0': 'index'})
small_df = df.set_index(["title", "section", "subsection", "p_number", "index"])
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
    prompt_sample, _ = construct_prompt(question, doc_embeddings2, small_df)
    print(f'Time for Constructing Prompt: {time.time() - start}')

    response = openai.Completion.create(
                    prompt=prompt_sample,
                    **COMPLETIONS_API_PARAMS
                )

    print(response["choices"][0]["text"].strip(" \n"))
## a lot of this is taken from the OpenAI cookbook
import numpy as np
import openai
import pandas as pd
import csv
import pickle

#storing private key in a file that won't be synced to github
from config import OPENAI_API_KEY, EMBEDDING_MODEL, COMPLETIONS_MODEL

from openai_functions import compute_doc_embeddings

openai.organization = "org-h2tLuOD0WsmSH4extTGzgOXU" #this is identical to other organization keys. this is fine to share
openai.api_key = OPENAI_API_KEY
openai.Model.list()

#need to start by reading the big DF first
df = pd.read_csv('FULL_DATA_new.csv')

#we want a unique numbered index for each paragraph in the entire dataset
df = df.rename(columns={'Unnamed: 0': 'index'})
# this will be the key that the dictionary returns
# df = df.set_index(["title", "section", "subsection", "p_number", "index"])


# Here is an example. I will ask a question about abduction, and a small section of the dataset
# will be embedded. The most similar 5 paragraphs will be appended to the question as context. 


small_df = df.iloc[:1000] #this will just have the first 30 rows
small_df = small_df.set_index(["title", "section", "subsection", "p_number", "index"])

doc_embeddings = compute_doc_embeddings(small_df)
with open('data/first_X_embeddings.pkl', 'wb') as f:
    pickle.dump(doc_embeddings, f)


print('done')
print('all done')

#NEED TO SAVE THE DOC EMBEDDINGS!

# OUT is a dictionary, with complex keys:
# out['Abduction', '1. Abduction: The General Idea', '1.2 The ubiquity of abduction', 8]
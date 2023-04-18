## a lot of this is taken from the OpenAI cookbook
import numpy as np
import openai
import pandas as pd

#storing private key in a file that won't be synced to github
from config import OPENAI_API_KEY 

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"


openai.organization = "org-h2tLuOD0WsmSH4extTGzgOXU" #this is identical to other organization keys. this is fine to share
openai.api_key = OPENAI_API_KEY
openai.Model.list()


df = pd.read_csv('first_1000.csv')

df['idx'] = [None] * len(df)


# print(f"{len(df)} rows in the data.")
# print(df.sample(5))

# test_string = "You happen to know that Tim and Harry have recently had a terrible row that ended their friendship. Now someone tells you that she just saw Tim and Harry jogging together. The best explanation for this that you can think of is that they made up. You conclude that they are friends again."
# test_embed = get_embedding(test_string)
# print(len(test_embed))

small_df = df.iloc[:30] #this will just have the first 5 rows 
small_df['idx'] = [None] * len(small_df)


prev = None
curr = ""
index = 0

for _, r in df.iterrows():

    curr = r.subsection

    #added str() to account for NAN since nan != nan
    if str(curr) == str(prev):
        index +=1
        r.idx = index
        
    else:
        index = 0
        r.idx = index

    prev = curr
    # print(r.text)

df = df.set_index(["title", "section", "subsection", "idx"])

pass
pass

def get_embedding(text: str, model: str=EMBEDDING_MODEL):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame): #-> dict[tuple[str, str], list[float]]
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.text) for idx, r in df.iterrows()
    }

def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

doc_embeddings = compute_doc_embeddings(df[:30])


print(doc_embeddings)

# OUT is a dictionary, with complex keys:
# out['Abduction', '1. Abduction: The General Idea', '1.2 The ubiquity of abduction', 8]



order_document_sections_by_query_similarity("Hilary Putnam’s book Reason, Truth, and History", doc_embeddings)[:5]



pass
pass


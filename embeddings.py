## a lot of this is taken from the OpenAI cookbook
import numpy as np
import openai
import pandas as pd
import csv
import pickle

#storing private key in a file that won't be synced to github
from config import OPENAI_API_KEY, EMBEDDING_MODEL, COMPLETIONS_MODEL

from openai_functions import compute_doc_embeddings, order_document_sections_by_query_similarity, construct_prompt

openai.organization = "org-h2tLuOD0WsmSH4extTGzgOXU" #this is identical to other organization keys. this is fine to share
openai.api_key = OPENAI_API_KEY
openai.Model.list()

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": .0,
    "max_tokens": 500,
    "model": COMPLETIONS_MODEL,
}


df = pd.read_csv('FULL_DATA_new.csv')

#we want a unique numbered index for each paragraph in the entire dataset
df = df.rename(columns={'Unnamed: 0': 'index'})
# this will be the key that the dictionary returns
# df = df.set_index(["title", "section", "subsection", "p_number", "index"])


# Here is an example. I will ask a question about abduction, and a small section of the dataset
# will be embedded. The most similar 5 paragraphs will be appended to the question as context. 

run_example = True
load_embeddings = True
if run_example:

    small_df = df.iloc[:50] #this will just have the first 30 rows
    small_df = small_df.set_index(["title", "section", "subsection", "p_number", "index"])


    if load_embeddings:
        
        with open('saved_dictionary.pkl', 'rb') as f:
            doc_embeddings = pickle.load(f)
    else:
        doc_embeddings = compute_doc_embeddings(small_df)
        with open('saved_dictionary.pkl', 'wb') as f:
            pickle.dump(doc_embeddings, f)
            

    pass
    pass

    #NEED TO SAVE THE DOC EMBEDDINGS!

    # OUT is a dictionary, with complex keys:
    # out['Abduction', '1. Abduction: The General Idea', '1.2 The ubiquity of abduction', 8]

    question = "Explain abduction, and how it is different than induction or deduction."
    #print the most similar ones

    # out_example = order_document_sections_by_query_similarity(question, doc_embeddings)[:5]
    # print(out_example)

    prompt = construct_prompt(question, doc_embeddings, small_df)

    #SAMPLE PROMPT:
    prompt_sample = """'Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don\'t know."\n\nContext:\n\n*  Abduction is normally thought of as being one of three major types of inference, the other two being deduction and induction. The distinction between deduction, on the one hand, and induction and abduction, on the other hand, corresponds to the distinction between necessary and non-necessary inferences. In deductive inferences, what is inferred is necessarily true if the premises from which it is inferred are true; that is, the truth of the premises guarantees the truth of the conclusion. A familiar type of example is inferences instantiating the schema\n*  A noteworthy feature of abduction, which it shares with induction but not with deduction, is that it violates monotonicity, meaning that it may be possible to infer abductively certain conclusions from a subset of a set S of premises which cannot be inferred abductively from S as a whole. For instance, adding the premise that Tim and Harry are former business partners who still have some financial matters to discuss, to the premises that they had a terrible row some time ago and that they were just seen jogging together may no longer warrant you to infer that they are friends again, even if—let us suppose—the last two premises alone do warrant that inference. The reason is that what counts as the best explanation of Tim and Harry’s jogging together in light of the original premises may no longer do so once the information has been added that they are former business partners with financial matters to discuss.\n\n Q: Explain abduction, and how it is different than induction or deduction.\n A:'"""
    print("===\n", prompt)


prompt_sample = """'Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don\'t know."\n\nContext:\n\n*  Abduction is normally thought of as being one of three major types of inference, the other two being deduction and induction. The distinction between deduction, on the one hand, and induction and abduction, on the other hand, corresponds to the distinction between necessary and non-necessary inferences. In deductive inferences, what is inferred is necessarily true if the premises from which it is inferred are true; that is, the truth of the premises guarantees the truth of the conclusion. A familiar type of example is inferences instantiating the schema\n*  A noteworthy feature of abduction, which it shares with induction but not with deduction, is that it violates monotonicity, meaning that it may be possible to infer abductively certain conclusions from a subset of a set S of premises which cannot be inferred abductively from S as a whole. For instance, adding the premise that Tim and Harry are former business partners who still have some financial matters to discuss, to the premises that they had a terrible row some time ago and that they were just seen jogging together may no longer warrant you to infer that they are friends again, even if—let us suppose—the last two premises alone do warrant that inference. The reason is that what counts as the best explanation of Tim and Harry’s jogging together in light of the original premises may no longer do so once the information has been added that they are former business partners with financial matters to discuss.\n\n Q: Explain abduction, and how it is different than induction or deduction.\n A:'"""


response = openai.Completion.create(
                prompt=prompt_sample,
                **COMPLETIONS_API_PARAMS
            )

print(response["choices"][0]["text"].strip(" \n"))

result_example = """Abduction is a type of inference that is non-necessary, meaning that the truth of the premises does not guarantee the truth of the conclusion. It is different from induction and deduction in that it violates monotonicity, meaning that it may be possible to infer certain conclusions from a subset of a set of premises which cannot be inferred from the set as a whole."""
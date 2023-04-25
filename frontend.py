import openai
import pandas as pd
import pickle
import streamlit as st

# st.write("whats up idk")

# x = st.slider('x')  # 👈 this is a widget
# st.write(x, 'squared is', x * x)

from config import OPENAI_API_KEY, COMPLETIONS_MODEL

from openai_functions import construct_prompt

@st.cache_data
def load_params():
    openai.organization = "org-h2tLuOD0WsmSH4extTGzgOXU" #this is identical to other organization keys. this is fine to share
    openai.api_key = OPENAI_API_KEY
    openai.Model.list()

    COMPLETIONS_API_PARAMS = {
        # We use temperature of 0.0 because it gives the most predictable, factual answer.
        "temperature": .0,
        "max_tokens": 500,
        "model": COMPLETIONS_MODEL,
    }

    return COMPLETIONS_API_PARAMS

@st.cache_data
def load_dataset():
    #LOAD THE EMBEDDINGS FROM A PKL FILE
    path_pkl = "data/first_1000_embeddings.pkl"
    with open(path_pkl, 'rb') as f:
        doc_embeddings = pickle.load(f)

    # NEED TO LOAD the dataset so that the paragraphs can be returned with the dictionary
    df = pd.read_csv('data/FULL_DATA_short.csv')
    df = df.rename(columns={'Unnamed: 0': 'index'})
    small_df = df.iloc[:1000] #this will just have the first X rows
    small_df = small_df.set_index(["title", "section", "subsection", "p_number", "index"])

    return doc_embeddings, small_df

@st.cache_data
def answer_question(question: str) -> str:
    prompt_sample, extra_info = construct_prompt(question, doc_embeddings, small_df)

    response = openai.Completion.create(
                    prompt=prompt_sample,
                    **COMPLETIONS_API_PARAMS
                )

    out = response["choices"][0]["text"].strip(" \n")
    return out, extra_info
# START THE Q&A PROCESS

COMPLETIONS_API_PARAMS = load_params()
doc_embeddings, small_df = load_dataset()

# question = "Please compare abduction to Bayesian Confirmation Theory"

question_from_website = st.text_input("Ask a Philosophy related question! ")

question, context = answer_question(question_from_website)
st.write(question)


st.write("Here's what we referenced:")
# st.write(pd.DataFrame({
#     'first column': ["Article", "Section", "Subsection", "Paragraph Number"],
#     'secasdffd column': [context[0][0], context[0][1], context[0][2], context[0][3]]
# }))

df = pd.DataFrame(context)
df.columns = ["Article", "Section", "Subsection", "Paragraph Number", "idx"]
st.write(df)

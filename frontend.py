import openai
import pandas as pd
import pickle
import streamlit as st
from PIL import Image


# Dictionary where the OpenAI API key is:
# /Users/johngearig/.streamlit/secrets.toml

# from config import OPENAI_API_KEY, COMPLETIONS_MODEL
COMPLETIONS_MODEL = "text-davinci-003"

from openai_functions import construct_prompt

@st.cache_data
def load_params(temp_in):
    openai.organization = "org-h2tLuOD0WsmSH4extTGzgOXU" #this is identical to other organization keys. this is fine to share
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    openai.Model.list()

    COMPLETIONS_API_PARAMS = {
        # We use temperature of 0.0 because it gives the most predictable, factual answer.
        "temperature": temp_in,
        "max_tokens": 500,
        "model": COMPLETIONS_MODEL
    }

    return COMPLETIONS_API_PARAMS

@st.cache_data
def load_dataset():
    #LOAD THE EMBEDDINGS FROM A PKL FILE
    path_pkl = "data/first_100_embeddings.pkl"
    with open(path_pkl, 'rb') as f:
        doc_embeddings = pickle.load(f)

    # NEED TO LOAD the dataset so that the paragraphs can be returned with the dictionary
    df = pd.read_csv('data/FULL_DATA_short.csv')
    df = df.rename(columns={'Unnamed: 0': 'index'})

    ##need to change this one
    small_df = df.iloc[:100] #this will just have the first X rows
    small_df = small_df.set_index(["title", "shorturl", "section", "subsection", "p_number", "index"])

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

# temp = st.slider('temperature', 0.0, 1.0)
temp=0

COMPLETIONS_API_PARAMS = load_params(temp)
doc_embeddings, small_df = load_dataset()

st.title("Philosophy GPT")
st.caption("A Q&A tool trained on the Stanford Encyclopedia of Philosophy, and using OpenAI's GPT3 completion API")
st.divider()

# question = "Please compare abduction to Bayesian Confirmation Theory"
# question_from_website = "Please compare abduction to Bayesian Confirmation Theory"

##DEBUG
question_from_website = st.text_input("Ask a philosophy related question! ",max_chars=200)
# question_from_website = "what is abduction?"

sep_url = "https://plato.stanford.edu/entries/"


if len(question_from_website) > 0:
    answer, context = answer_question(question_from_website)
    st.write(answer)
    st.divider()
    st.write("Here's what was referenced:")
    df = pd.DataFrame(context)
    df.columns = ["Article", "shorturl", "Section", "Subsection", "Paragraph Number", "idx"]


    #display all of the articles used in Context
    st.write(df.iloc[:,[0,1,2,3,4]])

    # generate hyperlinks 
   
    urls = df["shorturl"].unique()
    st.write("Links to source articles:")
    for url in urls:
        st.markdown(f"""[SEP article about {url.capitalize()}]({sep_url}{url})
                    """)
    
#display a landing picture
else:
    pic = Image.open("static/athens_zoomed.jpeg")
    st.image(pic)


## remove some of the stuff at the bottom.
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """ #can add: #MainMenu {visibility: hidden;}
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
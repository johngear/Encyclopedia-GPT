## WORK IN PROGRESS

I am writing software to prompt GPT (or any language model with a nice API) to answer Philosophy questions with reference to the Stanford Encyclopedia of Philosophy (one of the best online resources!). 

Currently, I am thinking that the work will be done with one of OpenAI's off-the-shelf completion models, where better prompting can act as a form of fine tuning. While fine tuning is possible, it is claimed to not improve factual accuracy, and is better suited for style transfer and other applicaitons..

By better prompting, I am basing much of this off the OpenAI cookbook (https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb), specifically, the pre-GPT3.5 push, which substantially changed some of the code. 

The basic workflow of the program is as follows:

A question asked to the interface will be EMBEDDED (meaning, a corresponding 1536 dimension vector is created from a paragraph of text), and compared each paragraph embedding in the entire encyclopedia. The most similar few paragraphs ought to have a lot of overlap with the question and will be appended to the question as CONTEXT, so that GPT can reference factual information within. We are assuming that the questions are answerable by concatenating and summarizing information found within SEP, but needs specific enough recall that the base models cannot find that level of specificity on their own.

This is what is being fed into the model:
"""Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\n Context: \n Question: \n Answer: \n """
    
The project took the form of a few stages.

1. A substantial one time data cleaning project to get the dataset into a usable form (read_pq_and_clean_pq.py)
2. Making a terminal-operated prototype with a partial dataset (backend.py, openai_functions.py)
3. Testing the prototype with a Streamlit Deployed Frontend (frontend.py)
4. Embedding the entire dataset with parallelization (parallel_api_call.py) and cleaning the output (read_jsonl.py)
5. (**CURRENT**) Fixing speed bottlenecks with the entire dataset (timing.py)

# Data Cleaning
read_pq_and_clean_pq.py is where I clean the data. This file is well commented. HuggingFace (https://huggingface.co/datasets/hugfaceguy0001/stanford_plato) has the entire SEP, but the part of the dataset that had the entire article text, was in a strange format. This was tedious. 

I split this and saved to a CSV, with each row having the title, section, subsection, paragraph #, full paragraph text, url and publish date. 

# Prototype:
Using the first 1000 paragraphs, you can prompt GPT via backend.py. This program sets up all of the OpenAI API calls, loads all of the data needed, constructs the prompt, and answers the question. It has some latency, taking a noticeable few seconds to answer, but reasonable IMO. 

Here is what it looks like: 

        Ask a philosophy related question! What is abduction?
        Selected 4 document sections:
        ('Abduction', '2. Explicating Abduction', nan, 1, 22)
        ('Abduction', '1. Abduction: The General Idea', '1.2 The ubiquity of abduction', 6, 18)
        ('Abduction', '1. Abduction: The General Idea', '1.1 Deduction, induction, abduction', 0, 4)
        ('Abduction', '2. Explicating Abduction', nan, 3, 24)



        Abduction is normally thought of as being one of three major types of inference, the other two being deduction and induction. The distinction between deduction, on the one hand, and induction and abduction, on the other hand, corresponds to the distinction between necessary and non-necessary inferences.

I consider this successful!

# Streamlit

I wanted a frontend that can be easily deployed to a website, and doesn't take a lot of focus. I had seen another project using Streamlit, and figured it was worth a shot. In frontend.py, it was easy to get the function of the backend working with a simple UI, and notably, being clear about what documents it was referencing to the user to reduced hallucinations. 

# Embeddings:
First, using the OpenAI API, I computed ~1141 embeddings (each embeddings is for 1 paragraph) to see how much this costs. OpenAI allegedly does not do Embeddings that much better than others, but at a low cost, it's worth it for the simplicity. This part cost 5 cents

The full, cleaned dataset is a CSV file with 164,300 rows, where each row is a paragraph. Then: It SHOULD cost $6 for the full embedding of the dataset, which is great! **UPDATE** It cost $10, which is a more than expected, but reasonable.

Unfortunately, the embedding process is pretty slow, although only needs to be done once. 1000 paragraphs took 5 minutes! Meaning, the total time could be upwards of 13 hours to get all the embeddings. Doing some digging, OpenAI has a program called api_request_parallel_processor.py which speeds up the requests rapidly. Still, it took approximately 2 hours for it to complete. 

# Speed Issues
This is the most daunting task now. I haven't really optimized for speed thus far, but with all of the dataset, it is substantially slower to the point where it isn't usable presently. OpenAI docs suggest the use of dictionaries as I have. Right now there is the issue of loading in the embeddings (30 seconds!).

Loading embeddings into dictionary: 28.3 seconds
Loading dataframe of all data:      1.8 seconds

I seems like the primary problem is with the dictionary ds, which I'm working on finding a fix for.
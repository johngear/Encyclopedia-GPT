## WORK IN PROGRESS

I am writing software to prompt GPT (or any language model with a nice API) to answer Philosophy questions with reference to the Stanford Encyclopedia of Philosophy (one of the best online resources!). 

Will update as the architecture takes shape, but for now, most of my effort has been in cleaning some very strangely formatted data.

Currently, I am thinking that the work will be done with the base model, using better prompting as opposed to fine tuning. Perhaps, later, I can use fine-tuning, as it wouldn't be that difficult to adjust.

By better prompting, I am basing much of this off the OpenAI cookbook (https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb), specifically, the pre GPT3.5 push, which changed a lof ot the code.

A question asked to the interface will be EMBEDDED, and compared each paragraph embedding in the entire SEP. The most similar few will be appended to the question as CONTEXT. Additionally the completion model will be instructed to say "I don't know" if it cannot answer.

This is based on the theory that questions are likely answerable by concatenating and summarizing information found within SEP, but is specific enough that the base models cannot recall the level of specificity. 

# Data Cleaning

read_pq_and_clean_pq.py is where I clean the data. This file is well commented. HuggingFace (https://huggingface.co/datasets/hugfaceguy0001/stanford_plato) has the entire SEP, but the part of the dataset that had the entire article text, was in a strange format. This was tedious. 

I split this and saved to a CSV, with each row having the title, section, subsection, paragraph #, full paragraph text, url and publish date. 

# Embeddings Cost:

First, using the OpenAI API, I computed ~1141 embeddings (each embeddings is for 1 paragraph) to see how much this costs. OpenAI allegedly does not do Embeddings that much better than others, but at a low cost, it's worth it for the simplicity.

It cost 5 cents

The full, cleaned dataset is a CSV file with 164,300 rows, where each row is a paragraph. Then: It SHOULD cost $6 for the full embedding of the dataset, which is great! I was worried this would be like $100.

Unfortunately, the embedding process is pretty slow, although only needs to be done once. 1000 paragraphs took 5 minutes! Meaning, the total time could be upwards of 13 hours to get all the embeddings :/
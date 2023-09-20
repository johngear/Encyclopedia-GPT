![alt text](https://github.com/johngear/Encyclopedia-GPT/blob/main/static/athens_zoomed.jpeg)

## You can use the tool at https://philosophy-chat.com 
Recent changes are in private repos for deployment.

I am writing software to prompt GPT (or any language model with a nice API) to answer Philosophy questions with reference to the Stanford Encyclopedia of Philosophy-- one of the best online resources!. 

Currently, this is working with one of OpenAI's off-the-shelf completion models, "text-davinci-003", where better prompting can give us better answers than fine tuning can. But doesn't give better factual recall from large datasets, which is our goal here.

By better prompting, I am basing much of this off the OpenAI cookbook (https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb), specifically, the pre-GPT3.5 push, which substantially changed some of the code. 

The basic workflow of the program is as follows:

A question asked to the interface will be EMBEDDED (meaning, a corresponding 1536 dimension vector is created from a paragraph of text using "text-embedding-ada-002"), and compared each paragraph embedding in the entire encyclopedia. The most similar few paragraphs ought to have a lot of overlap with the question and will be appended to the question as CONTEXT, so that GPT can reference factual information within. We are assuming that the questions are answerable by concatenating and summarizing information found within SEP, but needs specific enough recall that the base models cannot find that level of specificity on their own. Further, ChatGPT insists on giving answers to questions with lots of background information, and tries to output a handful of paragraphs. Whereas a student of philosophy may be more curious about sections of the SEP to reference, and would rather have a shorter, more correct answer.

Still, ChatGPT is quite good at philosophy-related questions, and it will be hard to output a better result. But we try regardless...

This is what is being fed into the model:
        """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\n Context: \n Question: \n Answer: \n """
    
The project took the form of a few stages.

1. A substantial one time data cleaning project to get the dataset into a usable form (`read_pq_and_clean_pq.py`)
2. Making a terminal-operated prototype with a partial dataset (`backend.py`, `openai_functions.py`)
3. Testing the prototype with a Streamlit Deployed Frontend (`frontend.py`)
4. Embedding the entire dataset with parallelization (parallel_api_call.py) and cleaning the output (`read_jsonl.py`)
5. Fixing speed bottlenecks with the entire dataset by adding a better search function (`UPDATED_openai_functions.py`, `timing.py`)
6. (**CURRENT**) Fixing Streamlit runtime speed (currently taking 5x as long as terminal)
7. (**CURRENT**) Looking at different scalable cloud providers and spinning up some VMs with Docker

# Data Cleaning
`read_pq_and_clean_pq.py` is where I clean the data. This file is well commented. HuggingFace (https://huggingface.co/datasets/hugfaceguy0001/stanford_plato) has the entire SEP, but the part of the dataset that had the entire article text, was in a strange format. This was tedious. 

I split this and saved to a CSV, with each row having the title, section, subsection, paragraph #, full paragraph text, url and publish date. 

#TODO There is probably a better way to do this, where paragraphs are concatenated together if they are short, as to avoid single-line paragraphs and account for the variety of author's writing styles.

# Prototype:
Using the first 1000 paragraphs, you can prompt GPT via backend.py. This program sets up all of the OpenAI API calls, loads all of the data needed, constructs the prompt, and answers the question. It has some latency, taking a noticeable few seconds to answer, but reasonable IMO. This is using the off-the-shelf `openai_functions.py`, which are slow due to dictionary usage and slow sorting algorithms. 

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

I wanted a frontend that can be easily deployed to a website, and doesn't take a lot of focus. I had seen another project using Streamlit, and figured it was worth a shot. In frontend.py, it was easy to get the function of the backend working with a simple UI, and notably, being clear about what documents it was referencing to the user to reduce hallucinations. This serves as a page where people can tackle a philosophical question with sources and a brief answer, rather than an oracle.

# Embeddings:
First, using the OpenAI API, I computed ~1141 embeddings (each embeddings is for 1 paragraph) to see how much this costs. OpenAI allegedly does not do Embeddings that much better than others, but at a low cost, it's worth it for the simplicity. This part cost 5 cents

The full, cleaned dataset is a CSV file with 164,300 rows, where each row is a paragraph. Then: It SHOULD cost $6 for the full embedding of the dataset, which is great! **UPDATE** It cost $10, which is a more than expected, but reasonable.

Unfortunately, the embedding process is pretty slow, although only needs to be done once. 1000 paragraphs took 5 minutes! Meaning, the total time could be upwards of 13 hours to get all the embeddings. Doing some digging, OpenAI has a program called api_request_parallel_processor.py which speeds up the requests rapidly. Still, it took approximately 2 hours for it to complete. `parallel_api_call.py` is 

# Speed Issues
This is the most daunting task now. I haven't really optimized for speed thus far, but with all of the dataset, it is substantially slower to the point where it isn't usable presently. OpenAI docs suggest the use of dictionaries as I have. Right now there is the issue of loading in the embeddings (30 seconds!).

Loading embeddings into dictionary: 28.3 seconds
Loading dataframe of all data:      1.8 seconds

I seems like the primary problem is with the dictionary data structure, which I'm working on finding a fix for.

### Fix!
The issue was twofold: first, with loading the embeddings, then with finding the most similar embeddings to append as context. 

For loading the embeddings, saving them as a massive pickled numpy array was an easy fix that I had always thought I was going to implement. For the vector search, the functions provided by OpenAI were sorting every element in the dictionary of embeddings by it's dot product similarity and returning the whole dictionary. This is immensely slow-- first we don't need to return all of the embeddings, just a handful of the most relevant. And second, there is no need to calculate the dot product for 160,000 vectors every time. Instead, I used FAISS (facebook ai similarity search) which excels in high dimension vector searching. `faiss.IndexFlatIP()` creates an index of the dataset and allows to search based on Inner Product (dot product, or cosine similarity when normalized to 1, as our data is). Then, I had to trace these changes and their repercussions through the program and created UPDATED_openai_functions.py which includes the big changes.

Now, the dataset loads in ~3 seconds, prompts are constructed in ~4 seconds, and the API call takes another ~2 seconds. Overall, this isn't lightning fast, but seems within the acceptable range of use for a specialized Q&A tool.


### Fixing on the Frontend

Streamlit is still super slow!

15 seconds to load dataset.
35 seconds to answer a question.

# Sample Prompt

This is what is actually being asked to the completions API

        Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."

        Context:

        * In its simplest form, abduction can best be described with  Peirce’s  1903 schema (Hartshorne & Weiss 1934: CP 5.189):
        * Like induction, and unlike deduction, abduction is not necessarily truth-preserving: in the example above, it is still possible that the defendant is not guilty after all, and that some other, unexpected phenomena caused the evidence to emerge. But abduction is significantly different from induction in that it does not only concern the generalization of prior observation for prediction (though it may also involve statistical data): rather, abduction is often backward-looking in that it seeks to explain something that has already happened. The key notion is that of bringing together apparently independent phenomena or events as explanatorily and/or causally connected to each other, something that is absent from a purely inductive argument that only appeals to observed frequencies. Cognitively, abduction taps into the well-known human tendency to seek (causal) explanations for phenomena (Keil 2006).
        * To this, Stathis Psillos (1999, Ch. 4) has responded by invoking a distinction credited to Richard Braithwaite, to wit, the distinction between premise-circularity and rule-circularity. An argument is premise-circular if its conclusion is amongst its premises. A rule-circular argument, by contrast, is an argument of which the conclusion asserts something about an inferential rule that is used in the very same argument. As Psillos urges, Boyd’s argument is rule-circular, but not premise-circular, and rule-circular arguments, Psillos contends, need not be viciously circular (even though a premise-circular argument is always viciously circular). To be more precise, in his view, an argument for the reliability of a given rule R that essentially relies on R as an inferential principle is not vicious, provided that the use of R does not guarantee a positive conclusion about R’s reliability. Psillos claims that in Boyd’s argument, this proviso is met. For while Boyd concludes that the background theories on which scientific methodology relies are approximately true on the basis of an abductive step, the use of abduction itself does not guarantee the truth of his conclusion. After all, granting the use of abduction does nothing to ensure that the best explanation of the success of scientific methodology is the approximate truth of the relevant background theories. Thus, Psillos concludes, Boyd’s argument still stands.
        * But not all inferences are of this variety. Consider, for instance, the inference of “John is rich” from “John lives in Chelsea” and “Most people living in Chelsea are rich.” Here, the truth of the first sentence is not guaranteed (but only made likely) by the joint truth of the second and third sentences. Differently put, it is not necessarily the case that if the premises are true, then so is the conclusion: it is logically compatible with the truth of the premises that John is a member of the minority of non-rich inhabitants of Chelsea. The case is similar regarding your inference to the conclusion that Tim and Harry are friends again on the basis of the information that they have been seen jogging together. Perhaps Tim and Harry are former business partners who still had some financial matters to discuss, however much they would have liked to avoid this, and decided to combine this with their daily exercise; this is compatible with their being firmly decided never to make up.
        * There have been proposals dealing with these issues. One of them is by Pacuit, Parikh, & Cogan (2006), which uses a setting in which actions can be considered “good” or “bad”. It introduces a notion of knowledge-based obligation under which an agent is obliged to perform an action \(\alpha\) if and only if \(\alpha\) is an action which the agent can perform and she knows that it is good to perform \(\alpha\). This is then a form of absolute obligation which remains until the agent performs the required action.

        Q: What is abduction?
        A: 
        
Answer:
        Abduction is a form of inference that seeks to explain something that has already happened. It involves bringing together apparently independent phenomena or events as explanatorily and/or causally connected to each other. Cognitively, abduction taps into the well-known human tendency to seek (causal) explanations for phenomena.

import pandas as pd
# import pyarrow

import ast

#this is the dataset i downloaded from Hugging Face
full_data = pd.read_parquet('sep_data.parquet')

#Removes almost all of the columns. we are left with URL, Title, publication date, and Main Text. Discarding the rest for now
trimmed_data = full_data.drop(columns=["preamble","toc", "bibliography","related_entries"])

#Create an EMPTY dataframe, that will hold the following information. This is what we will save the cleaned data into.
cleaned_data = pd.DataFrame(columns=["title","shorturl", "pubinfo", "section", "subsection", "text"])


#trimmed_data is formatted as the following:
# shorturl: which can just be added to the SEP URL and link
# title: a string
# pubinfo: a string that is a sentence about its publication date. This would be easy to get date format from
# main_text: COMPLEX FORMAT THAT WE NEED TO PARSE. 
# trimmed_data.loc[0, 'main_text'] returns an example of the format
# this returns AN ARRAY, with an element for every main section of the article.
# inside each array, is another array which corresponds to the subsection.s


#SAMPLE: Abduction
# 1. Abduction: The General Idea
# 1.1 Deduction, induction, abduction
# 1.2 The ubiquity of abduction
# 2. Explicating Abduction
# 3. The Status of Abduction
# 3.1 Criticisms
# 3.2 Defenses
# 4. Abduction versus Bayesian Confirmation Theory

i = 0
index = 0
for index, row in trimmed_data.iterrows():

    index+=1
    print(index)

    current_row_full_text = row['main_text']

    for val in current_row_full_text:
        # print(val["section_title"])


        #this should add all the paragraphs that fall under primary headings (1., 2. for example)
        #BUT, will not add things in subheadings, like 1.1, 2.2 for example
        for paragraph in val['main_content']:
            

            new = [row['title'], 
                   row['shorturl'], 
                   row['pubinfo'], 
                   val['section_title'], 
                   None, 
                   paragraph.replace('\n', ' ')]

            cleaned_data.loc[i] = new
            i+=1
            
        
        if len(val['subsections']) > 0:
            for subsection in val['subsections']:
                for paragraph in subsection['content']:
                    new = [row['title'], 
                        row['shorturl'], 
                        row['pubinfo'], 
                        val['section_title'], 
                        subsection['subsection_title'], 
                        paragraph.replace('\n', ' ')]
                    
                    cleaned_data.loc[i] = new
                    i+=1


    # if i >1000:
    #     break

pass
# cleaned_data = cleaned_data.set_index(["title", "section"])

# pd.set_option('display.max_columns', None)
# print(cleaned_data.head())


cleaned_data.to_csv('FULL_DATA.csv', index=False)
print('done')

pass
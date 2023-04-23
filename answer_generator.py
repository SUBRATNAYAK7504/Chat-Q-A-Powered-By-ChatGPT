
import pandas as pd
import tiktoken


import openai
from openai.embeddings_utils import distances_from_embeddings


tokenizer = tiktoken.get_encoding("cl100k_base")

openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""

VERBOSE_MODE = True
MAX_TOKENS = 500


def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    
    return serie

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, page_num, file_name, url, doc_type = "pdf", max_tokens = MAX_TOKENS, splitter = ". "):

    # Split the text into sentences
    sentences = text.split(splitter)

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    if doc_type == "pdf":
        page_num_list = []
        file_name_list = []
    elif doc_type == "url":
        url_list = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            if doc_type == "pdf":
                page_num_list.append(page_num)
                file_name_list.append(file_name)
            elif doc_type == "url":
                url_list.append(url)
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1
    
    if doc_type == "pdf":
        return chunks, page_num_list, file_name_list
    elif doc_type == "url":
        return chunks, url_list
def preprocess_data(out_text, doc_type = "pdf"):

    if doc_type == "url":
        df = pd.DataFrame(out_text, columns = ["url", "text"])
    elif doc_type == "pdf":
        df = pd.DataFrame(out_text, columns = ["page_num", "text", "file_name"])

    df['text'] = remove_newlines(df.text)
    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    shortened = []

    if doc_type == "pdf":
        out_page_num_list = []
        out_file_name_list = []
    elif doc_type == "url":
        out_url_list = []

    # Loop through the dataframe
    for i,row in df.iterrows():

        # If the text is None, go to the next row
        if row['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row['n_tokens'] > MAX_TOKENS:
            if doc_type == "pdf":
                chunks, page_num_list, file_name_list = split_into_many(row['text'], row["page_num"], row["file_name"], None)
                out_page_num_list.extend(page_num_list)
                out_file_name_list.extend(file_name_list)
            elif doc_type == "url":
                chunks, url_list = split_into_many(row['text'], None, None, row["url"])
                out_url_list.extend(url_list)

            shortened.extend(chunks)
        
        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( row['text'] )
            if doc_type == "pdf":
                out_page_num_list.append(row["page_num"])
                out_file_name_list.append(row["file_name"])
            elif doc_type == "url":
                out_url_list.append(row["url"])
    if doc_type == "pdf":
        df = pd.DataFrame({"text": shortened, "page_num": out_page_num_list, "file_name": out_file_name_list})
    elif doc_type == "url":
        df = pd.DataFrame({"text": shortened, "url": out_url_list})

    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    return df

def create_context(
    question, df, max_len=1000, size="ada", doc_type = "pdf"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    source_of_answer = []
    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])
        if doc_type == "pdf":
            source_of_answer.append((str(row["file_name"]).split()[1], row["page_num"]))
        if doc_type == "url":
            source_of_answer.append((row["url"]))

    # Return the context

    source_sent = "Answer is from:"

    for i, tup in enumerate(source_of_answer):
        if i!= 0:
            source_sent += ", "+str(tup)
        else:
            source_sent += " "+str(tup)


    return "\n\n###\n\n".join(returns), source_sent

def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None,
    doc_type = "pdf"
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context, source = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
        doc_type=doc_type
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt = f"Answer the question polite and formally based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip(), source
    except Exception as e:
        print(e)
        return ""




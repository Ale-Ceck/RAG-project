#embedding function

import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

def get_embeddings(): #(chunks):
    embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small"
    )
    return embeddings

import argparse
import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from get_embedding_function import get_embeddings

load_dotenv(find_dotenv()) # read local .env file

CHROMA_PATH = os.getenv("CHROMA_PATH")

openai.api_key  = os.environ['OPENAI_API_KEY']

llm_name = "gpt-4o-mini"
llm = ChatOpenAI(
    model_name = llm_name,
    temperature = 0,
)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    #Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Query text")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    embedding_function = get_embeddings()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    results = db.similarity_search_with_score(query_text, k = 5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, question = query_text)
    response_text = llm.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text
    
if __name__ == "__main__":
    main()
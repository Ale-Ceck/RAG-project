import argparse
import os
import shutil
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document  
from get_embedding_function import get_embeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv(find_dotenv())
DATA_PATH = os.getenv("DATA_PATH")
CHROMA_PATH = os.getenv("CHROMA_PATH")
max_batch_size = 166

def main():

    # Check if database need to be cleared
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    # Create (or update) the data store
    documents = load_documents()
    chunks  = split_documents(documents)
    batches = []
    batches = create_batches(chunks=chunks, batch_size=max_batch_size)
    for batch in batches:
        add_to_chroma(batch)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()
    
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        is_separator_regex= False,
    )
    return text_splitter.split_documents(documents)

def create_batches(chunks, batch_size):
    for i in range(0, len(chunks), batch_size):
        yield chunks[i: i + batch_size]

def add_to_chroma(chunks: list[Document]):
    #Load existing database
    db = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = get_embeddings()
    )

    #Calculate page ID
    chunks_with_ids = calculate_chunks_ids(chunks)

    #Add or update the documents
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in current DB: {len(existing_ids)}")

    #Add elements not present in the DB yet
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunks_ids(chunks:list[Document]):
    last_page_id = None
    current_chunk_index = 0 
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        #Use an index for multiple chunks on the same page
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
    
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
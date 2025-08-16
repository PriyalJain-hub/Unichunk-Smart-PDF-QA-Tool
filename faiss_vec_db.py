from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
import json
import pickle

# Load Config
def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

#Method to save embeddings to FAISS Vector DB
def save_embeddings_to_faiss(docs, db_name="summary_store"):
    
    # embedding_model = HuggingFaceEmbeddings(
    #     model_name="BAAI/bge-base-en-v1.5",
    #     encode_kwargs={"normalize_embeddings": True}  # important for cosine similarity
    # )

    # load the local model
    print("Saving embeddings to faiss db: summary_store")
    embedding_model = HuggingFaceEmbeddings(
        model_name="local_bge_base_en_v1_5",
        encode_kwargs={"normalize_embeddings": True}
    )
    summary_store = FAISS.from_documents(
            documents=docs,
            embedding=embedding_model
    )

    summary_store.save_local(f"{db_name}")

def main():
    config=load_config()
    OUTPUT_DIR=config["output_dir"]
    with open(f"{OUTPUT_DIR}/summarized_docs.pkl", "rb") as f:
        summarized_docs = pickle.load(f)

    save_embeddings_to_faiss(summarized_docs)

if __name__ == "__main__":
    main()


    
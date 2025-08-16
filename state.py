# state.py - to store common variables that need to be accessible to multiple .py files
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
        

# load the local model
embedding_model = HuggingFaceEmbeddings(
        model_name="local_bge_base_en_v1_5",
        encode_kwargs={"normalize_embeddings": True}
)

#summary_store
summary_store = FAISS.load_local("./summary_store", embeddings=embedding_model, allow_dangerous_deserialization=True)

#all_chunks
config=load_config()
OUTPUT_DIR=config['output_dir']
with open(f"{OUTPUT_DIR}/all_chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)

processed_dbs = {}
choices_list = [
    "2024_Report_on_the_State_of_Cybersecurity_in_the_Union.pdf",
    "Medical_Device_Coordination_Group_Document.pdf",
    "shifting_gears_2025_automotive_cybersecurity_report.pdf",
    "Tech_Trends_Report.pdf",
    "All Files"
]
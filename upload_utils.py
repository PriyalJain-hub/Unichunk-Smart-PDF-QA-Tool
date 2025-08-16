from pathlib import Path
from datetime import datetime
import shutil
from parsing import process_pdf_to_markdown
from chunking import process_all_markdowns_and_metadata
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from state import processed_dbs

#Passing new pdf through ingestion pipeline
def process_new_pdf(local_path, base_output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_pdf_dir = Path(base_output_dir) / f"session_{timestamp}"
    new_pdf_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(local_path).stem
    output_md = new_pdf_dir / f"{stem}.md"
    output_meta = new_pdf_dir / f"{stem}_metadata.json"

    print(f"Processing {local_path.name}")
    process_pdf_to_markdown(local_path, str(output_md), str(output_meta), model_name="gpt-4o-mini")
    all_docs, all_chunks = process_all_markdowns_and_metadata(new_pdf_dir)

    embedding_model = HuggingFaceEmbeddings(
        model_name="local_bge_base_en_v1_5",
        encode_kwargs={"normalize_embeddings": True}
    )

    temp_db = FAISS.from_documents(
        documents=all_chunks,
        embedding=embedding_model
    )
    return temp_db

#Handle file upload - when new file is uploaded within the tool - to chat
def handle_file_upload(file, query):
    if file is None:
        return "No file uploaded."

    upload_dir = Path("./temp_uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = Path(file.name)
    file_name = file_path.name
    local_path = upload_dir / file_name

    shutil.copy(file_path, local_path)
    new_pdf_dir = "./new_pdf_dir"
    temp_db = process_new_pdf(local_path, new_pdf_dir)

    processed_dbs[file_name] = temp_db

    return f"File '{file_name}' uploaded and processed. Ask me anything!"

import os
import json
from parsing import process_pdf_to_markdown
from chunking import process_all_markdowns_and_metadata
from summarization import summarize_pages
from faiss_vec_db import save_embeddings_to_faiss
import pickle
from pathlib import Path

#Method to run parsing on all input files
def call_parse(data_dir, parse_output_dir, captioning_model):
    parse_output_dir = Path(parse_output_dir)
    parse_output_dir.mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, filename)
            output_md = parse_output_dir / f"{Path(pdf_path).stem}.md"
            output_meta = parse_output_dir / f"{Path(pdf_path).stem}_metadata.json"
            process_pdf_to_markdown(pdf_path, str(output_md), str(output_meta), captioning_model)

#Method to process all pdfs through ingestion - parse,chunk,store in db
def process_all_pdfs(data_dir, parse_output_dir, output_dir, captioning_model):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    call_parse(data_dir, parse_output_dir, captioning_model)
    all_docs, all_chunks = process_all_markdowns_and_metadata(parse_output_dir)

    print(f"Parsed and chunked {len(all_docs)} pages with total {len(all_chunks)} chunks.")
    summarized_docs = summarize_pages(all_docs)
    with open(os.path.join(output_dir, "all_docs.pkl"), "wb") as f:
        pickle.dump(all_docs, f)
    with open(os.path.join(output_dir, "all_chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)
    with open(os.path.join(output_dir, "summarized_docs.pkl"), "wb") as f:
        pickle.dump(summarized_docs, f)
        
    save_embeddings_to_faiss(summarized_docs)
    
            
def process_new_pdf(file, new_pdf_dir):
    new_pdf_dir = Path(new_pdf_dir)
    new_pdf_dir.mkdir(parents=True, exist_ok=True)
    output_md = new_pdf_dir / f"{Path(file.name).stem}.md"
    output_meta = new_pdf_dir/ f"{Path(file.name).stem}_metadata.json"
    print(f"Processing {file.name}")
    process_pdf_to_markdown(file.name, str(output_md), str(output_meta))
    all_docs, all_chunks = process_all_markdowns_and_metadata(new_pdf_dir)
    save_embeddings_to_faiss(all_chunks,"temp_db")


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    process_all_pdfs(config["data_dir"], config['parse_output_dir'] ,config["output_dir"], config["model"])

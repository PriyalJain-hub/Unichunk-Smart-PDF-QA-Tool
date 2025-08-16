import re
from typing import List, Dict
from langchain_core.documents import Document
import nltk
from nltk.tokenize import sent_tokenize
import os
import json
import pickle



nltk.download('punkt')
nltk.download('punkt_tab')

# Load Config
def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)
    
#Methods to do semantic, layout aware chunking
def count_tokens(text: str) -> int:
    return len(text.split())

def is_heading(line: str) -> bool:
    return bool(re.match(r"^#+\s", line.strip()))

#Method to chunk each page into multiple chunks - semantic and layout aware - and add required metadata
def chunk_page(doc: Document, CHUNK_TOKEN_LIMIT=500, OVERLAP_SENTENCES=2) -> List[Document]:
    text = doc.page_content
    metadata = doc.metadata.copy()
    source_file = metadata.get("source", "unknown_file")
    page = metadata.get("page", "unknown_page")

    #split text into lines to avoid line breaks
    lines = text.split('\n')
    chunks = []
    buffer = []
    buffer_tokens = 0
    chunk_id = 0
    current_section = None

    #If theres a heading or a image or a page split - start a new chunk
    for line in lines:
        if is_heading(line) or "[IMAGE_PLACEHOLDER:" in line or line.strip() == "---":
            if buffer:
                chunk_text = '\n'.join(buffer).strip()
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_ref": f"{source_file}_p{page}_c{chunk_id}",
                        "source": source_file,
                        "page": page,
                        "chunk_no": chunk_id,
                        "section_title": current_section,
                        "saliency": None
                    }
                ))
                chunk_id += 1
                buffer = []
                buffer_tokens = 0

        if is_heading(line):
            current_section = line.strip()

        #If adding another sentence will make chunk size greater than token limit - break chunk
        sentences = sent_tokenize(line)
        for sent in sentences:
            sent_tokens = count_tokens(sent)
            if buffer_tokens + sent_tokens > CHUNK_TOKEN_LIMIT:
                chunk_text = '\n'.join(buffer).strip()
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_ref": f"{source_file}_p{page}_c{chunk_id}",
                        "source": source_file,
                        "page": page,
                        "chunk_no": chunk_id,
                        "section_title": current_section,
                        "saliency": None
                    }
                ))
                chunk_id += 1
                buffer = buffer[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
                buffer_tokens = sum(count_tokens(s) for s in buffer)

            buffer.append(sent)
            buffer_tokens += sent_tokens
    #If theres something left in buffer in the end - add as a new chunk
    if buffer:
        chunk_text = '\n'.join(buffer).strip()
        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                "chunk_ref": f"{source_file}_p{page}_c{chunk_id}",
                "source": source_file,
                "page": page,
                "chunk_no": chunk_id,
                "section_title": current_section,
                "saliency": None
            }
        ))

    return chunks

#Creating a lookup to map image placeholders in chunks to image data in json
#to store image data (base64) as chunk metadata
def load_image_metadata_lookup(metadata_json_path):
    with open(metadata_json_path, "r") as f:
        image_metadata = json.load(f)

    image_lookup = {}
    for page_meta in image_metadata:
        for img in page_meta["images"]:
            image_lookup[img["placeholder"]] = img
    return image_lookup

#to store image data (base64) as chunk metadata
def inject_image_metadata_into_chunks(chunks, image_lookup):
    for chunk in chunks:
        content = chunk.page_content
        found_placeholders = re.findall(r"\[IMAGE_PLACEHOLDER:[^\]]+\]", content)
        chunk.metadata["images"] = [image_lookup[ph] for ph in found_placeholders if ph in image_lookup]
    return chunks

#to add saliency score to metadata
def inject_saliency_scores(chunks):
    for chunk in chunks:
        content = chunk.page_content if hasattr(chunk, "page_content") else chunk.get("page_content", "")
        saliency = 0

        if re.search(r"^\s*#\s", content, re.MULTILINE):
            saliency = max(saliency, 3)
        if re.search(r"^\s*##\s", content, re.MULTILINE):
            saliency = max(saliency, 2)
        if re.search(r"\*\*[^*]+\*\*", content):
            saliency = max(saliency, 1)

        # Ensure metadata exists and preserve existing keys
        if not hasattr(chunk, "metadata") or chunk.metadata is None:
            chunk.metadata = {}

        chunk.metadata["saliency"] = saliency if saliency > 0 else None

    return chunks


#split each page - save (all_docs), send each page for chunking - save returned chunks(all_chunks)
def process_all_markdowns_and_metadata(parse_output_dir, chunk_token_limit=500, overlap_sentences=2):
    print("Chunking for two tier db:")
    all_chunks = []
    all_docs= []

    for file in os.listdir(parse_output_dir):
        if file.endswith(".md"):
            base_name = file[:-3]  # remove .md
            #Reading parsed outputs
            md_path = os.path.join(parse_output_dir, f"{base_name}.md")
            meta_path = os.path.join(parse_output_dir, f"{base_name}_metadata.json")

            if not os.path.exists(meta_path):
                print(f"[!] Skipping {base_name}, no metadata found.")
                continue

            # Load markdown and metadata
            with open(md_path, "r", encoding="utf-8") as f:
                markdown_text = f.read()
            #Loading image, metadata lookup
            image_lookup = load_image_metadata_lookup(meta_path)

            # Split by pages
            pages = markdown_text.split("\n\n---\n\n")
            docs = [
                Document(page_content=page, metadata={"source": f"{base_name}.pdf", "page": i + 1})
                for i, page in enumerate(pages)
            ]

            all_docs.extend(docs)

            # Chunk & inject metadata
            for doc in docs:
                chunks= chunk_page(doc, chunk_token_limit, overlap_sentences)  # Your chunking function
                chunks_with_meta = inject_image_metadata_into_chunks(chunks, image_lookup)
                chunks_with_score = inject_saliency_scores(chunks_with_meta)
                all_chunks.extend(chunks_with_score)

    return all_docs, all_chunks


# Entrypoint
def main():
    config = load_config()
    PARSE_OUTPUT_DIR = config["parse_output_dir"]
    OUTPUT_DIR=config["output_dir"]
    all_docs, all_chunks= process_all_markdowns_and_metadata(PARSE_OUTPUT_DIR, config["CHUNK_TOKEN_LIMIT"],config["OVERLAP_SENTENCES"])
    #Saving lists - all_docs, all_chunks, summarized_docs

    with open(f"{OUTPUT_DIR}/all_docs.pkl", "wb") as f:
        pickle.dump(all_docs, f)

    with open(f"{OUTPUT_DIR}/all_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

if __name__ == "__main__":
    main()
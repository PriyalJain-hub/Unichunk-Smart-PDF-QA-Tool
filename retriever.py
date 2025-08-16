from collections import defaultdict
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from state import all_chunks, embedding_model


#Method to append characters at the start and end of chunks - to increase context awareness
def stitch_chunks_with_neighbors(top_chunks: list[Document], all_chunks: list[Document], prev_chars=300, next_chars=300):
    combined_result = []
    meta_store = []

    for main_chunk in top_chunks:
        meta = main_chunk.metadata
        source = meta.get("source")
        page = meta.get("page")
        section_title = meta.get("section_title")
        images = meta.get("images")

        meta_store.append({
            "source": source,
            "page": page,
            "section_title": section_title,
            "images": images
        })

        chunk_no = meta.get("chunk_no")

        same_page_chunks = [
            c for c in all_chunks if c.metadata.get("source") == source and c.metadata.get("page") == page
        ]
        same_page_chunks.sort(key=lambda c: c.metadata.get("chunk_no", 0))

        main_index = next((idx for idx, c in enumerate(same_page_chunks) if c.metadata.get("chunk_no") == chunk_no), None)

        if main_index is None:
            print(f"[ERROR] Could not find chunk_no {chunk_no} in source {source} page {page}")
            continue

        parts = []
        if main_index > 0:
            parts.append(same_page_chunks[main_index - 1].page_content[-prev_chars:])
        parts.append(same_page_chunks[main_index].page_content)
        if main_index < len(same_page_chunks) - 1:
            parts.append(same_page_chunks[main_index + 1].page_content[:next_chars])

        stitched_text = "\n".join(parts)
        combined_result.append(stitched_text)

    final_text = "\n\n---\n\n".join(combined_result)
    return final_text, meta_store

#to perform two tier similarity search on vector db
def deep_chunk_similarity_search(query: str, selected_file='all files', top_k: int = 5):
    from state import summary_store
        
    summary_matches = []
    #Metadata filter - when one file is selected - similarity search on embeddings of same file- Top K Page Summaries retrieved - from selected file
    try:
        if selected_file.lower() != "all files":

            print(f"Selected filter source: '{selected_file}'")
            print("Similarity Search on Selected File Summaries:")
            top_summary_matches = summary_store.similarity_search(query, k=top_k, filter={"source": selected_file})
            print(top_summary_matches[:1])
        else:
            #Top K Page Summary Retrieval on all embeddings
            top_summary_matches = summary_store.similarity_search(query, k=top_k)

        summary_matches.extend(top_summary_matches)
    except Exception as e:
        print(f"[Summary similarity search failed] {e}")
        return None, None

    #Use metadata from retrieved summaries - source and page number - to get all chunks from those retrieved pages
    #Then perform another top k similarity search on those fetched chunks
    final_contexts = []
    for summary_doc in summary_matches:
        source = summary_doc.metadata.get("source")
        page = summary_doc.metadata.get("page")
        print("Finding chunks from top k pages:")
        filtered_chunks = [
            chunk for chunk in all_chunks
            if chunk.metadata.get("source") == source and chunk.metadata.get("page") == page
        ]
       
        if filtered_chunks:
            final_contexts.extend(filtered_chunks)
        else:
            print(f"[ERROR] No chunks found for {source} p{page}")
            continue
        
    if not final_contexts:
        print("[ERROR] No matching chunks for any top-k summary pages.")
        return None, None


    final_contexts_with_ids = [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in final_contexts
    ]

    if not final_contexts_with_ids:
        print("[ERROR] No documents to build FAISS index from.")
        return None, None

    print("Top K Filtering - Tier 2 - on fetched chunks")
    filtered_chunk_store = FAISS.from_documents(
        documents=final_contexts_with_ids,
        embedding=embedding_model
    )

    top_filtered_matches = filtered_chunk_store.similarity_search(query, k=10)
    final_combined_text, meta_store = stitch_chunks_with_neighbors(top_filtered_matches, all_chunks)
    return final_combined_text, meta_store

#Formatting metadata - to also render image if it is present in the retrieved chunks
def format_metadata(meta_store):
    print("Formatting metadata:")
    grouped = defaultdict(lambda: {"pages": set(), "images": []})
    for item in meta_store:
        grouped[item['source']]["pages"].add(item['page'])
        if "images" in item:
            grouped[item['source']]["images"].extend(item["images"])

    formatted_blocks = []
    for source, data in grouped.items():
        sorted_pages = sorted(data["pages"])
        pages_str = ", ".join(map(str, sorted_pages))
        block = f"**Source**: {source}\n**Pages**: {pages_str}"

        for img in data["images"]:
            caption = img.get("caption", "No caption available.")
            img_b64 = img.get("image_base64")
            if img_b64:
                img_tag = f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%; border:1px solid #ccc; margin-top:10px;" />'
                block += f"\n\n {img_tag}\n\n{caption}"

        formatted_blocks.append(block)

    return "\n\n---\n\n".join(formatted_blocks)

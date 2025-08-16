from openai import OpenAI
from langchain.schema import Document
from tqdm import tqdm
import os
import json
import pickle
from dotenv import load_dotenv

load_dotenv()

# Load Config
def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

client = OpenAI(
        api_key=os.getenv("GPT4O_API_KEY"),
        base_url=os.getenv("GPT4O_BASE_URL"),
        default_headers={"genaiplatform-farm-subscription-key": os.getenv("GPT4O_HEADER_KEY")}
    )

#summarize each page (all_docs) and save (summarized_docs)
def summarize_page(text: str, system_prompt: str = "You are a summarization assistant that generates structured, detailed summaries to create embeddings for accurate retrieval.", model = "gpt-4o-mini") -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarize the following page:\n\n{text.strip()}"}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_query={"api-version": "2024-08-01-preview"}
    )
    return response.choices[0].message.content.strip()

def summarize_pages(pages: list[Document]) -> list[Document]:
    print("Summarizing all pages:")
    summarized_docs = []
    for page in tqdm(pages, desc="Summarizing pages"):
        if not page.page_content.strip():
            continue
        summary = summarize_page(page.page_content)
        summarized_docs.append(Document(
            page_content=summary,
            metadata={**page.metadata, "summary_of_page": True}
        ))
    return summarized_docs


# Entrypoint
def main():
    config = load_config()
    all_docs=[]
    OUTPUT_DIR=config["output_dir"]
    model=config["model"]
    with open(f"{OUTPUT_DIR}/all_docs.pkl", "rb") as f:
        all_docs = pickle.load(f)
    summarized_docs= summarize_pages(all_docs)
    #Saving lists - summarized_docs

    with open(f"{OUTPUT_DIR}/summarized_docs.pkl", "wb") as f:
        pickle.dump(summarized_docs, f)

if __name__ == "__main__":
    main()
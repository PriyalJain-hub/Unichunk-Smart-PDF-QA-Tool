import os
from collections import defaultdict
from langchain.schema import Document
from upload_utils import processed_dbs
from retriever import deep_chunk_similarity_search
from retriever import format_metadata
from openai import OpenAI
from dotenv import load_dotenv
from state import choices_list
import re
import tiktoken


load_dotenv()

client = OpenAI(
    api_key=os.environ["GPT4O_API_KEY"],
    base_url=os.environ["GPT4O_BASE_URL"],
    default_headers={"genaiplatform-farm-subscription-key": os.environ["GPT4O_HEADER_KEY"]}
)

model = "gpt-4o-mini"

MAX_TOKENS = 128000

# Use encoder for the model
enc = tiktoken.encoding_for_model(model)

def estimate_tokens(messages):
    return sum(len(enc.encode(m["content"])) for m in messages)

#based on the delected file - it does similarity search - retrieval and fetches final text to send to LLM for response
def generate_response_from_combined_text(query: str, selected_file: str, history, system_prompt: str = None, model="gpt-4o-mini"):

    meta_store = None
    meta_file = None
    
    if not selected_file:
        history.append(("System", "Please upload or select a file first."))
        return history, ""

    if selected_file in processed_dbs:
        temp_db = processed_dbs[selected_file]
        top_matches = temp_db.similarity_search(query, k=3)
        #print(top_matches)

        text = "\n".join([doc.page_content for doc in top_matches])
        img_tag=""
        for data in top_matches:
            for img in data.metadata.get("images", []):
                caption = img.get("caption", "No caption available.")
                img_b64 = img.get("image_base64")
                if img_b64:
                    img_tag = f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%; border:1px solid #ccc; margin-top:10px;" />'
                    img_tag = re.sub(r'[\ud800-\udfff]', '', img_tag)
    
        meta_file = selected_file + img_tag
    
    elif selected_file in choices_list:
        print("Going into deep_chunk_similarity_search")
        text, meta_store = deep_chunk_similarity_search(query, selected_file)

        #If similarity search fails, query routed to LLM with context, if no history - exits with a system message
        if not text or not meta_store:
            print("[Fallback Triggered] No relevant content retrieved.")

            if history:
                fallback_context = history[-1][1]  # Last assistant response
                text = fallback_context
                if any(phrase in fallback_context.lower() for phrase in [
                    "i cannot process", 
                    "i can't analyze", 
                    "i don't have access to", 
                    "as a language model", 
                    "i do not have the ability"
                ]):
                    history.append((query, "Please ask a question."))
                    return history, ""
                # Clean fallback text before reuse - because img base64 strings can bloat the context limit
                cleaned_fallback = re.sub(r'<img[^>]*>', '', fallback_context)
                cleaned_fallback = re.sub(r'\*\*Information about the Data Retrieved\*\*:.*?\*\*Overall Summary\*\*:', '', cleaned_fallback, flags=re.DOTALL)
                cleaned_fallback = re.sub(r'\s+', ' ', cleaned_fallback).strip()

                # Heuristic: reject if <100 characters OR <20 alphabetic words
                word_count = len(re.findall(r'\b[a-zA-Z]{3,}\b', cleaned_fallback))
                if len(cleaned_fallback) < 100 or word_count < 20:
                    history.append((query, "The previous response doesn't contain enough meaningful information to elaborate on. Please ask a more specific question."))
                    return history, ""
                else:
                    text = cleaned_fallback
                    meta_store = [{
                        "source": selected_file,
                        "page": "context from previous response",
                        "section_title": "Fallback",
                        "images": []
                    }]
            else:
                history.append((query, "No relevant information found. Please ask a more specific question."))
                return history, ""
    else:
        chat_history = chat_history + [("System", f"File '{selected_file}' not processed. Please upload again.")]
        return chat_history, ""
    # print(text)

    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant.\n\n"
            "Please generate a **detailed, thorough, and well-explained** answer based ONLY on the following source text. "
            "Do NOT add any new information or paraphrase the meaning. Strictly preserve the original wording and facts.\n"
            "Your answer should fully address the user's question, providing examples or explanations as needed."
        )

    # Build message list including history
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Add previous conversation turns from history to messages
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    # Add source text context as a system/user message or assistant context
    messages.append(
        {"role": "system", "content": f"Source Text (for reference only):\n\n{text.strip()}"}
    )

    # Add current user query as the last user message
    messages.append({"role": "user", "content": f"{query}\n\nPlease provide a detailed, comprehensive answer based ONLY on the above source text. Explain all relevant points thoroughly."})

    estimated_tokens = estimate_tokens(messages)
    if estimated_tokens > MAX_TOKENS:
        history = []
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Source Text (for reference only):\n\n{text.strip()}"},
            {"role": "user", "content": f"{query}\n\nPlease provide a detailed, comprehensive answer based ONLY on the above source text. Explain all relevant points thoroughly."}
        ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_query={"api-version": "2024-08-01-preview"},
        temperature=0
    )

    answer = response.choices[0].message.content.strip()
    metadata_summary = (
        format_metadata(meta_store) if meta_store else
        meta_file if meta_file else
        "Sorry, could not obtain metadata for this file."
    )
    metadata_summary += "\n\n---\n\n"

    res = f"**Information about the Data Retrieved**:\n{metadata_summary}\n\n**Overall Summary**:\n{answer}"

    res = res.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")

    # Append new turn to history
    history = history + [(query, res)]

    # Return updated history (chatbot expects history), and clear input textbox
    return history, ""


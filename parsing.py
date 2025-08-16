import os
import json
import fitz
import base64
import numpy as np
import cv2
import imutils
import re
import pytesseract
from pathlib import Path
from pytesseract import Output
from llama_index.core.schema import Document
from llama_cloud_services import LlamaParse
import nltk
from dotenv import load_dotenv

load_dotenv()


nltk.download("words")
word_list = set(nltk.corpus.words.words())

pytesseract.pytesseract.tesseract_cmd = os.getenv("Tesseract_PATH")

# Load Config
def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

# Utils
#Method to extract images from page
def extract_images_from_page(page):
    images = []
    for _, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        base_image = page.parent.extract_image(xref)
        images.append((base_image["image"], base_image["ext"]))
    return images

#Method to encode image to base64
def encode_image_to_base64(cv2_image, img_format=".jpg"):
    success, buffer = cv2.imencode(img_format, cv2_image)
    if not success:
        raise ValueError("Image encoding failed")
    return base64.b64encode(buffer).decode("utf-8")

#Methods to calculate scores to check orientation and discard unnecessary images
def text_quality_score(text):
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    valid_words = [word for word in tokens if word in word_list]
    return len(valid_words) / len(tokens) if tokens else 0

def get_text_score(image):
    config = "--psm 6"
    text = pytesseract.image_to_string(image, config=config)
    length_score = len(text.strip())
    quality_score = text_quality_score(text)
    return length_score * quality_score, text

#Method to correct orientation
def auto_correct_orientation(image):
    best_score, best_text, best_image = 0, "", image
    for angle in [0, 90, 180, 270]:
        rotated = imutils.rotate_bound(image, angle)
        score, text = get_text_score(rotated)
        if score > best_score:
            best_score, best_text, best_image = score, text, rotated
    return best_text, best_image, best_score

# GPT-4o Captioning
def _gen_message_with_image(sysprompt, userprompt, image_b64):
    return [
        {"role": "system", "content": sysprompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": userprompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }
    ]
#Method to call LLM to get response
def _completion_with_image(image_b64, model_name):
    import openai
    client = openai.OpenAI(
        api_key=os.getenv("GPT4O_API_KEY"),
        base_url=os.getenv("GPT4O_BASE_URL"),
        default_headers={"genaiplatform-farm-subscription-key": os.getenv("GPT4O_HEADER_KEY")}
    )
    messages = _gen_message_with_image(
        "You are an image captioning AI",
        "Describe this image thoroughly and clearly. Mention all text, structure, and visual elements.",
        image_b64
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        extra_query={"api-version": "2024-08-01-preview"},
        temperature=0
    )
    return response.choices[0].message.content.strip()

# LlamaParse
def parse_with_llamaparse(pdf_path):
    parser = LlamaParse(
        api_key=os.getenv("LLAMAPARSE_API_KEY"),
        num_workers=4,
        result_type="markdown",
        premium_mode=True,
        verbose=True
    )
    return parser.load_data(Path(pdf_path))

# Main Parsing Logic/Function
def process_pdf_to_markdown(pdf_path, output_md_path, metadata_json_path, model_name="gpt-4o-mini"):
    try:
        print("Parsing:")
        #Parsing text , tables, charts with LlamaParse
        parsed_pages = parse_with_llamaparse(pdf_path)
    except Exception as e:
        raise ValueError(f"LlamaParse failed: {e}")
    
    doc = fitz.open(pdf_path)
    
    if len(parsed_pages) == 0:
        raise ValueError("LlamaParse returned 0 pages. Possible API issue or invalid file.")

    if len(parsed_pages) != len(doc):
        print(f"Warning: LlamaParse returned {len(parsed_pages)} pages, but PDF has {len(doc)} pages.")


    file_name = Path(pdf_path).name
    final_md, metadata_records = [], []

    for i, page in enumerate(doc):
        md_text = parsed_pages[i].text
        print("Image Processing:")
        #Extracting images using PyMuPDF
        images = extract_images_from_page(page)
        image_blocks = []

        for idx, (img_bytes, img_ext) in enumerate(images):
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            image_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            #Correcting orientation of image
            ocr_text, rotated_image, score = auto_correct_orientation(image_cv2)

            #If score less than threshold, do not save image
            if score > 150:
                base64_img = encode_image_to_base64(rotated_image)
                caption = _completion_with_image(base64_img, model_name)
                image_id = f"{file_name}_page{i+1}_img{idx}"
                placeholder = f"[IMAGE_PLACEHOLDER:{image_id}]"

                md_text += f"\n\n!{caption}\n\n{placeholder}"
                image_blocks.append({
                    "image_id": image_id,
                    "page": i + 1,
                    "caption": caption,
                    "ocr_text": ocr_text,
                    "image_base64": base64_img,
                    "image_ext": img_ext,
                    "placeholder": placeholder
                })

        final_md.append(md_text)
        metadata_records.append({
            "source_file": file_name,
            "page": i + 1,
            "images": image_blocks
        })

    #Writing .md and .json to folders
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(final_md))

    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2)

    print(f"Markdown: {output_md_path}")
    print(f"Metadata: {metadata_json_path}")

#Method to run parsing on a list of pdfs
def run_on_pdf_list(pdf_paths, output_dir, model_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in pdf_paths:
        try:
            output_md = output_dir / f"{Path(pdf_path).stem}.md"
            output_meta = output_dir / f"{Path(pdf_path).stem}_metadata.json"
            print(f"Processing {Path(pdf_path).name}")
            process_pdf_to_markdown(pdf_path, str(output_md), str(output_meta), model_name)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    print("All PDFs processed.")

# Entrypoint
def main():
    config = load_config()
    run_on_pdf_list(
        pdf_paths=config["pdf_paths"],
        output_dir=config["parse_output_dir"],
        model_name=config["model"]
    )

if __name__ == "__main__":
    main()

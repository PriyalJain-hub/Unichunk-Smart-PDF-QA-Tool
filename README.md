# UniChunk Smart PDF Q\&A Tool

UniChunk makes it simple to chat with your PDFs, summarize insights, and upload new files for instant querying. Follow this guide to get set up and running locally.

***

## Installation

### 1. Create a Virtual Environment

```bash
conda create --name unichunk_env
```


### 2. Activate the Environment

```bash
conda activate unichunk_env
```


### 3. Install Required Packages

Ensure you have a valid `requirements.txt` in your project directory:

```bash
pip install -r requirements.txt
```


***

## Environment Setup

Create a `.env` file in your project folder (or edit the existing one):

```env
LLAMAPARSE_API_KEY=<YOUR_API_KEY>
GPT4O_API_KEY="dummy"
GPT4O_BASE_URL=<URL>
GPT4O_HEADER_KEY=<YOUR_TOKEN>
Tesseract_PATH=<YOUR_PATH>
```

- Use existing keys or add your personal ones.
- For a new `LLAMAPARSE_API_KEY`:
Log in to Llama Cloud → create account → generate API Key.
- **`Tesseract_PATH`** should point to your `tesseract.exe` file.
Tesseract OCR **must be installed.**


### Tesseract Installation

#### Windows

Download [Windows-64bit Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) → Run installer → Add to `PATH`, and set path in `.env`.
*Already in PATH?*
Comment out line 23 in `parsing.py`:

```python
# pytesseract.pytesseract.tesseract_cmd = os.getenv("Tesseract_PATH")
```


#### Linux/Azure

```bash
sudo apt-get install tesseract-ocr-eng
```


***

## Configuration

Replace the sample input files and paths in `config.json` with your own:

```json
{
    "pdf_paths": [
      "data/2024 Report on the State of Cybersecurity in the Union.pdf",
      "data/Tech_Trends_Report.pdf"
    ],
    "data_dir": "./data",
    "output_dir": "./ingestion_output",
    "new_pdfs_dir": "./new_pdfs_dir",
    "parse_output_dir": "./parse",
    "model": "gpt-4o-mini",
    "CHUNK_TOKEN_LIMIT": 500,
    "OVERLAP_SENTENCES": 2
}
```

- Input PDFs go into `data/`
- `output_dir` stores processed docs
- New uploads and parsed files will be saved in respective folders

***

## Running the Tool

Launch the app locally:

```bash
python -m main
```

A localhost server will provide a link to access the UniChunk Q\&A Tool.

***

## Project Structure

| Folder/File | Description |
| :-- | :-- |
| `data/` | Raw input PDFs |
| `ingestion_output/` | Processed `.pkl` files: full docs, chunks, summaries |
| `local_bge_base-en_v1_5/` | Embedding model |
| `new_pdfs_dir/` | Parsed files for newly uploaded PDFs |
| `notebooks/` | Parsing \& ingestion pipelines (`.ipynb`) |
| `parse/` | Parsed output of PDFs |
| `summary_store/` | Vector database with summarized-doc embeddings |
| `temp_uploads/` | Storage for new uploads |
| `.py files` | Modular scripts: can run independently or together |
| `main.py` | Entrance: launches local UniChunk UI |


***

## Features

- **Chat with PDFs:** Talk to your uploaded documents—question and summarize with ease.
- **Upload \& Parse:** Add new files, get instant parsing and querying.
- **Flexible Selection:** Pick any PDF from the dataset or your uploads for chat.

***

**Note:**
Sample data is not provided due to file size; use your own files for testing.

***

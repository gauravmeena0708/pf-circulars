# PDF Document RAG Query System

## Overview

This project implements a Retrieval Augmented Generation (RAG) system designed to answer questions about a collection of PDF documents. It processes PDFs (including image-based ones with tables), extracts text and table content using OCR and table detection models, builds a searchable vector index, and then uses a Large Language Model (LLM) to generate answers based on the most relevant retrieved information from your documents.

The system provides both a command-line interface (`main.py`) for processing and querying, and a Streamlit web application (`app.py`) for a more interactive experience.

## Features

* **PDF Parsing**: Handles text-based and image-based PDFs.
* **Table Extraction**: Detects and extracts text from tables within PDFs.
* **OCR Integration**: Uses EasyOCR for text extraction from images and scanned portions of PDFs.
* **Vector Indexing**: Creates a FAISS vector index from document chunks for efficient similarity search.
* **Modular Design**: Code is structured into logical modules for parsing, indexing, retrieval, and answer generation.
* **Configurable**: Key parameters (model names, paths, API keys) are managed through a configuration file and environment variables.
* **Two Interfaces**:
    * `main.py`: A command-line tool for batch processing, indexing, and querying.
    * `app.py`: A Streamlit web application for interactive use.
* **Caching**: Streamlit app caches loaded models and index data for improved performance.

## Project Structure

The project is organized into the following Python modules:

* `config.py`: Handles all configurations, including API keys and model parameters.
* `pdf_parser.py`: Converts PDFs to images, detects tables, and performs OCR to extract structured content.
* `vector_indexer.py`: Creates, saves, and loads the FAISS vector index from processed PDF content.
* `retriever.py`: Retrieves relevant text chunks from the index based on a user query.
* `answer_generator.py`: Formats prompts and interacts with an LLM to generate answers.
* `main.py`: Command-line interface to orchestrate the RAG pipeline.
* `app.py`: Streamlit web application for an interactive user interface.
* `requirements.txt`: Lists all Python dependencies.
* `.env` (user-created): For storing sensitive API keys locally.

## Prerequisites

* Python 3.8+
* An active Hugging Face account and an API Token (`HF_TOKEN`) for accessing LLMs on the Hugging Face Hub.
* System dependencies for `pdf2image` (like `poppler-utils` on Linux):
    ```bash
    sudo apt-get update && sudo apt-get install -y poppler-utils
    ```
    (For other OS, please refer to `pdf2image` documentation.)

## Setup Instructions

1.  **Clone the Repository (if applicable)**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**:
    Create a file named `.env` in the root directory of the project. Add your Hugging Face API token to this file:
    ```env
    HF_TOKEN="your_actual_huggingface_api_token"
    # NGROK_AUTH_TOKEN="your_ngrok_auth_token" # If you plan to use ngrok with Streamlit
    ```
    Replace `"your_actual_huggingface_api_token"` with your real token.

## Running the Application

You can interact with the RAG system using either the command-line interface or the Streamlit web application.

### 1. Command-Line Interface (`main.py`)

The `main.py` script is used for processing a directory of PDFs, building/loading an index, and answering queries from the terminal.

**Usage Examples:**

* **Process PDFs, build/load index, and ask a query:**
    ```bash
    python main.py /path/to/your/pdf_directory --query "What is the main topic discussed?"
    ```

* **Force re-processing and re-indexing of PDFs:**
    (Useful if PDF content has changed or you want to rebuild the index from scratch)
    ```bash
    python main.py /path/to/your/pdf_directory --reindex
    ```

* **Specify a custom directory for storing the index:**
    ```bash
    python main.py /path/to/your/pdf_directory --query "A specific question" --index_dir /custom/path/for/index_files
    ```

* **View all command-line options:**
    ```bash
    python main.py --help
    ```

### 2. Streamlit Web Interface (`app.py`)

The `app.py` script launches a web application for a more interactive experience.

**To run the Streamlit app:**

1.  Ensure you are in the project's root directory and your virtual environment is activated.
2.  Run the following command:
    ```bash
    streamlit run app.py
    ```
3.  This will typically open the application in your default web browser (e.g., at `http://localhost:8501`).

**Using the Streamlit App:**

1.  **Enter PDF Directory Path**: In the sidebar, input the full path to the directory containing your PDF files.
2.  **Load and Process**: Click the "Load and Process PDF Directory" button. You can check "Force Re-index PDFs" if needed. The app will process the PDFs and build/load the vector index. Progress and status messages will be displayed.
3.  **Enter Query**: Once the directory is processed, a text input field will appear. Type your question about the documents.
4.  **View Answer**: The LLM's answer will be displayed, along with snippets from the source documents that were used as context.

## Configuration

* **General Settings**: Most application settings (default model names, paths, LLM parameters) are defined in `config.py`. You can modify these defaults if needed.
* **API Keys**: Sensitive API keys (like `HF_TOKEN`) should **only** be set in the `.env` file or as environment variables. They are loaded by `config.py`. **Do not hardcode API keys directly into Python files.**

## Potential Future Enhancements

* Support for other document types (e.g., .txt, .docx).
* More advanced chunking strategies for text.
* Sophisticated re-ranking of retrieved documents before sending to LLM.
* Asynchronous processing for the Streamlit app to handle long-running PDF processing tasks more gracefully.
* Option to select different embedding or LLM models from the UI.
* More robust error handling and user feedback.
* Integration with a database for persistent storage of processed data and metadata.

## Contributing

Contributions are welcome! Please feel free to fork the repository, make changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the `LICENSE` file (if you create one) for details.
# Document RAG Query Application

This application allows you to query documents using a Retrieval Augmented Generation (RAG) approach.
It can load a pre-built FAISS vector index or process PDF documents from a specified directory to build a new index.
Queries are then answered by retrieving relevant text chunks from the index and generating a response using a lightweight Language Model.

## Prerequisites

*   Python 3.8 - 3.11
*   Git

## Setup

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Hugging Face Token:**
    This application uses models from the Hugging Face Hub, which requires an API token.
    *   Obtain a token from [Hugging Face settings](https://huggingface.co/settings/tokens).
    *   You can set this token as an environment variable:
        ```bash
        export HF_TOKEN="your_hugging_face_token_here"
        ```
    *   Alternatively, create a `.env` file in the root of the project directory and add your token there:
        ```
        HF_TOKEN="your_hugging_face_token_here"
        ```
        The application will load this `.env` file using `python-dotenv` (which is in `requirements.txt`).

## Vector Index

The application is configured to first look for a pre-built FAISS index.

*   **Using the Pre-built Index:**
    *   Ensure your FAISS index files are located at:
        *   `vector_store/data_index/faiss_index.index`
        *   `vector_store/data_index/faiss_index.texts.json`
    *   When the app starts, it will attempt to load this index. Status messages will appear in the sidebar.

*   **Processing New PDF Documents:**
    *   If the pre-built index is not found, or if you wish to process a different set of documents:
        1.  Place your PDF files into a directory (e.g., a subdirectory under `data/`).
        2.  In the Streamlit application's sidebar, under "Process New PDF Directory (Optional)":
            *   Enter the path to your PDF directory.
            *   Optionally, check "Force Re-index PDFs" if you want to re-process even if a cached index for that directory exists.
            *   Click "Process PDF Directory".
        3.  This will create a new FAISS index specific to that directory (stored within `vector_store/app_<directory_name>_index/`). The application will then use this newly processed index for queries.

## Running Locally

1.  Ensure your virtual environment is activated and `HF_TOKEN` is set.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Deployment

### Streamlit Cloud

1.  **Push your code to a GitHub repository.**
2.  **Sign up or log in to [Streamlit Cloud](https://share.streamlit.io/).**
3.  **Click "New app" and connect your GitHub repository.**
    *   Select the repository and branch.
    *   The main script file should be `app.py`.
4.  **Advanced Settings:**
    *   Ensure the Python version matches your local environment (e.g., 3.9, 3.10).
    *   Go to the "Secrets" section and add your `HF_TOKEN`:
        ```
        HF_TOKEN="your_hugging_face_token_here"
        ```
5.  **Deploy!**

### Hugging Face Spaces

1.  **Push your code to a GitHub repository OR prepare to upload it directly.**
2.  **Sign up or log in to [Hugging Face](https://huggingface.co/).**
3.  **Create a new Space:**
    *   Click on your profile picture, then "New Space".
    *   Give your Space a name.
    *   Select "Streamlit" as the Space SDK.
    *   Choose "Docker" for the Space hardware. You can usually start with a free CPU instance.
    *   Choose whether to create from an existing GitHub repo or create a new empty Space to upload files to.
4.  **Configure the Space:**
    *   If creating from GitHub, ensure the correct branch and main application file (`app.py`) are specified.
    *   If creating an empty space, upload your files (`app.py`, `config.py`, `requirements.txt`, `vector_indexer.py`, `answer_generator.py`, `retriever.py`, `pdf_parser.py`, and your `vector_store` directory with the pre-built index).
        *   **Important for `vector_store`**: Due to typical storage limits on free tiers of HF Spaces, if your `vector_store/data_index` is very large, you might need to consider alternatives like Git LFS for the index files or a hosted vector database. For moderately sized indexes, direct upload should work.
    *   Ensure your `requirements.txt` is present and correct.
    *   Go to the "Settings" tab of your Space.
    *   Under "Repository secrets", add your `HF_TOKEN`:
        *   Secret name: `HF_TOKEN`
        *   Secret value: `your_hugging_face_token_here`
5.  **The Space should build and deploy automatically.** You'll be able to access it via `your-username-your-space-name.hf.space`.

---
This README provides comprehensive instructions for setting up, running, and deploying the application.
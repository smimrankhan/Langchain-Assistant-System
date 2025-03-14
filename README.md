# AI Knowledge Assistant

This repository contains an AI Knowledge Assistant application built using Streamlit and LangChain. The application leverages OpenAI's GPT-3.5-turbo model to answer questions based on provided documents.

## Features

- Retrieve and answer questions based on the provided context.
- Display document similarity search results.
- Contextual help and FAQ section.

## Setup and Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/smimrankhan/AI-Knowledge-Assistant.git
   cd AI-Knowledge-Assistant
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```plaintext
   OPEN_API_KEY=your_openai_api_key
   ```

## Usage

1. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

2. **Interact with the Application:**
   - Input your question in the provided text input box.
   - View the AI-generated response and document similarity search results.
   - Explore additional features like contextual help and FAQ.

### Author 
S. M. Imran Khan
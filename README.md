# LLM App

Demo project based on [Hands-On AI: Building LLM-Powered Apps](https://www.linkedin.com/learning/hands-on-ai-building-llm-powered-apps?trk=learning-course_related-content-card&upsellOrderOrigin=default_guest_learning) course.

## Run app

```bash
# Start an ollama server
ollama serve

# Activate venv in the project root and run:
chainlid run app/app.py
```

## Initial setup

```bash
# Setup python venv
mkdir 2025-llm-app
cd 2025-llm-app
uv init
uv add chainlit langchain langchain-community
```

## Notes

Main technologies used:
- `chainlid` - for GUI and chatbot environment
- `langchain` - for LLM integration and PDF processing
- `ollama` - for LLM server
- `chromadb` - for vector database

Calculating the chunk size when splitting text:
- Llama 3.1 (including the llama3.1-8B we are using) allows a context window of 128k tokens.
- The chunk size in RecursiveCharacterTextSplitter is in number of characters.
- The number of documents that the LLM will remember depends on the chunk size. E.g. if we set size to 25k, we can accomodate 5 chunks/documents.
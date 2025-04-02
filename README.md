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



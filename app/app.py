import chainlit as cl
from chainlit.types import AskFileResponse
from chromadb import EphemeralClient
from chromadb.config import Settings
from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema import StrOutputParser
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings


def create_search_engine(documents: list[Document], embedding: Embeddings) -> VectorStore:
    """Create vector store from langchain documents.

    Takes a list of Langchain Documents and an embedding model API wrapper
    and build a search index using a VectorStore.

    Parameters
    ----------
    docs : list[Document]
        List of Langchain Documents to be indexed into
        the search engine.

    embedding : Embeddings
        Encoder model API used to calculate embedding

    Returns
    -------
    search_engine : VectorStore
        Langchain VectorStore with the documents
    """
    client_settings = Settings(allow_reset=True, anonymized_telemetry=False)
    client = EphemeralClient(settings=client_settings)

    # Reset the search engine to ensure we don't use old copies
    search_engine = Chroma(client=client, client_settings=client_settings)
    search_engine._client.reset()

    search_engine = Chroma.from_documents(
        client=client,
        documents=documents,
        embedding=embedding,
        client_settings=client_settings,
    )

    return search_engine


def process_pdf(file: AskFileResponse) -> list[Document]:
    """Processes one PDF file from a Chainlit AskFileResponse object by first
    loading the PDF document and then chunk it into sub documents. Only
    supports PDF files.
    Args:
        file (AskFileResponse): input file to be processed

    Raises:
        ValueError: when we fail to process PDF files. We consider PDF file
        processing failure when there's no text returned. For example, PDFs
        with only image contents, corrupted PDFs, etc.

    Returns:
        list[Document]: List of Document(s). Each individual document has two
        fields: page_content(string) and metadata(dict).
    """
    if file.type != "application/pdf":
        raise TypeError("Only PDF files are supported")

    loader = PDFPlumberLoader(file.path)
    documents_raw = loader.load()

    # Split the pdf text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    documents = text_splitter.split_documents(documents_raw)

    if not documents:
        raise ValueError("PDF file parsing failed.")

    # Add an identifier
    for i, d in enumerate(documents):
        d.metadata["source"] = f"sorce_{i}"

    return documents


@cl.on_chat_start
async def on_chat_start():

    # 1. Process a pdf file and save the data in the user session
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to chat with...", accept=["application/pdf"], max_size_mb=20
        ).send()
    file = files[0]

    msg = cl.Message(content=f"Processing {file.name}")
    await msg.send()

    docs = process_pdf(file=file)
    cl.user_session.set("docs", docs)

    msg.content = f"{file.name} processed. Loading..."
    await msg.update()

    # 2. Index documents into the search engine
    embedding = OllamaEmbeddings(model="llama3.1")
    try:
        search_engine = await cl.make_async(create_search_engine)(documents=docs, embedding=embedding)
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
        raise SystemError
    msg.content = f"`{file.name}` loadded. You can now ask questions!"
    await msg.update()

    # 3. Create a prompt template and LLM chain
    model = ChatOllama(model="llama3.1")
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", "You are ollama, senior python developer."),
            ("human", "{question}"),
        ]
    )
    chain = LLMChain(llm=model, prompt=prompt, output_parser=StrOutputParser())

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message) -> None:
    chain = cl.user_session.get("chain")
    response = await chain.ainvoke(question=message.content, callbacks=[cl.LangchainCallbackHandler()])
    await cl.Message(content=response).send()

from tempfile import NamedTemporaryFile

import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.chains.llm import LLMChain
from langchain.chat_models.ollama import ChatOllama
from langchain.document_loaders import PDFPlumberLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
    response = await chain.arun(question=message.content, callbacks=[cl.LangchainCallbackHandler()])
    await cl.Message(content=response).send()

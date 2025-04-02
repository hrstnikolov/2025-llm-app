import chainlit as cl
from langchain.chat_models.ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema import StrOutputParser


@cl.on_chat_start
async def on_chat_start():
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
    response = await chain.arun(
        question=message.content, callbacks=[cl.LangchainCallbackHandler()]
    )
    await cl.Message(content=response).send()

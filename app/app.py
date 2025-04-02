import chainlit as cl


@cl.on_message
async def main(message: cl.Message) -> None:
    await cl.Message(content=message.content).send()

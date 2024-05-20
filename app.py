import chainlit as cl
from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)


def get_chunked_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    chunked_documents = text_splitter.split_documents(data)
    return chunked_documents


def get_template():
    template = """Answer the question based only on the following context:
        
        {context}
        
        Question: {question}
        """
    return template


@cl.on_chat_start
async def on_chat_start():
    store = LocalFileStore("./cache")  # TODO: How do we create a local file store to for our cached embeddings?
    embedder = CacheBackedEmbeddings.from_bytes_store(embedding_model, store)  # TODO: How do we create our embedder?
    loader = CSVLoader(file_path="./dataset/IMDB.csv")
    data = loader.load()
    print(len(data))
    vector_store = FAISS.from_documents(get_chunked_documents(data), embedder)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    # TODO: How do we save our vector store locally?
    vector_store.save_local("vector_store")
    retriever = vector_store.as_retriever()
    model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)
    prompt_template = ChatPromptTemplate.from_template(get_template())
    runnable = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt_template
                    | model
                    | StrOutputParser()
                )
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

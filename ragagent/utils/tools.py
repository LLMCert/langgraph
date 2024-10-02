from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader
import os

print("*********",os.getcwd())
print("++++++++",os.listdir(os.getcwd()))
loader = TextLoader("monk.txt")
loader.load()
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "RAG",
    "Search and return information about Monk Avvari, A swiss White Sheppard dog.",
)

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b
def add(a: int, b: int) -> int:
    """add a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b
def subtract(a: int, b: int) -> int:
    """subtract b from a.

    Args:
        a: first int
        b: second int
    """
    return a - b
tools = [retriever_tool,multiply,add,subtract]
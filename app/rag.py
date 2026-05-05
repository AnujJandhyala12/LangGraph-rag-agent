import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def create_retriever():
    # ✅ Load PDF
    loader = PyPDFLoader("data/quiz.pdf")
    docs = loader.load()

    # ✅ Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)

    # ✅ Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "rag-agent-index"

    # ✅ Create index only if it doesn't exist
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,        # all-MiniLM-L6-v2 outputs 384 dimensions
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    # ✅ Store chunks in Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        index_name=index_name
    )

    return vectorstore.as_retriever(search_kwargs={"k": 4})
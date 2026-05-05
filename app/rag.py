import os
import time
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from pinecone import Pinecone, ServerlessSpec


def create_retriever():
    # ✅ Load the generated credit risk report (not quiz.pdf)
    loader = TextLoader("data/credit_risk_report.txt")
    docs = loader.load()

    # ✅ Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ Semantic chunking — splits by meaning not character count
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85
    )
    split_docs = splitter.create_documents(
        [doc.page_content for doc in docs]
    )
    print(f"Created {len(split_docs)} semantic chunks")

    # ✅ Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "credit-risk-index"

    # ✅ Create index only if it doesn't exist
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)
        print(f"Created Pinecone index: {index_name}")

    # ✅ Store in Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        index_name=index_name
    )

    # ✅ MMR retrieval — relevance + diversity
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7}
    )
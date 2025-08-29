from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_data = load_pdf_files(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunk = text_split(filter_data)

embedding = download_hugging_face_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

index_name = "medical-chatbot"


if not pc.has_index (index_name):
    pc.create_index(
        name = index_name,
         dimension=384,
         metric="cosine",
         spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)


# Embed each chunk and upsert the embeddings into your pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    index_name=index_name,
    embedding=embedding,
)
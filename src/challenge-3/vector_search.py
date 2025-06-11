import os
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_iris import IRISVector

loader = JSONLoader(
    file_path='./data/tweets_all.jsonl',
    jq_schema='.note',
    json_lines=True
)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

username = '_SYSTEM'
password = 'SYS'
hostname = 'iris'
port = 1972
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

COLLECTION_NAME = "financial_tweets"

db = IRISVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

query = "How is Beyond Meat doing?"
docs_with_score = db.similarity_search_with_score(query)
for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)

learner_query="your search phrase"
docs_with_score = db.similarity_search_with_score(content)
print(docs_with_score[0])
from dotenv import load_dotenv
import os

load_dotenv("../.env")

os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")


# creating sample documents
from langchain_core.documents import Document

documents = [
    Document(
        page_content = "Dogs are great companions, known for their loyalty and friendliness.",
        metadata = {"source": "mammal-pets-doc"}
    ),
    Document(
        page_content = "Cats are independent pets that often enjoy their own space.",
        metadata = {"source": "mammal-pets-doc"}
    )
]


# **************
#* loading pdf
# **************
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path = "../data/Deep_Learning_A_Visual_Approach.pdf")

pages = loader.load()
print(len(pages))

print(pages[33].page_content)
print(pages[33].metadata)


# ***********
#* Splitting
# ***********
# so that the relevent portion of teh doc are not wasted out by surrounding text
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 250,
    add_start_index = True
)

all_splits = text_splitter.split_documents(pages)

print(len(all_splits))

print(all_splits[330].page_content)
print(all_splits[330].metadata)

# start_index = from where the chuck started (character position)
# doc --> pages --> split 
# so each page will split and page start from char=0 to its length
# so start_index can be same if page in metadata is different
# if page in metadata is same then start_index will be unique
# Orignal (page from pages) = 
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
# after splitting (chunk_size = 20, chunk_verap = 5) = 
# 00000000000000000000
#                00000000000000000000
#                               00000000000000000000
#                                              00000000000000000000
#                                                             00000000000000000000


# ***********
#* Embedding
# ***********
# vector representation of words, phrases or text
# map words, phrases or text to dense vector in lower dimensional space
# where semantically (meaning and context same) similar words are closer to each other
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding = GoogleGenerativeAIEmbeddings(
    model = "models/embedding-001"
)

# make some vector of splitted text as example
vector_0 = embedding.embed_query(all_splits[0].page_content)
vector_1 = embedding.embed_query(all_splits[1].page_content)

print(vector_0)
print(vector_1)

print(len(vector_0))
print(len(vector_1))

print(len(vector_0) == len(vector_1))


# ***************
#* Vector Store
# ***************
# After embedding, text ---converted into---> vector
# We will do the vector similarity search with the i/p on vector store
# vector similarity (ex. cosine similarity)

# in-memory
from langchain_core.vectorstores import InMemoryVectorStore
vector_store_inmemory = InMemoryVectorStore(embedding = embedding)

# mongodb
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")

client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_vectorstores"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "langchain_test_index_vectorstores"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

vector_store_mongodb = MongoDBAtlasVectorSearch(
    embedding = embedding,
    collection = MONGODB_COLLECTION,
    index_name = ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn = "cosine"
)

# using mongodb
ids = vector_store_mongodb.add_documents(documents = all_splits)

len(ids)
len(all_splits)

print(ids[0])

# similarity search
response = vector_store_mongodb.similarity_search(query = "learning", k = 2)
print(response)

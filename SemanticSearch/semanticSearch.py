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
ATLAS_VECTOR_SEARCH_INDEX_NAME = "langchain_test_index"
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
response = vector_store_mongodb.similarity_search(query = "deep", k = 5)
print(response)

# Print the page_content of each document from response
for i, doc in enumerate(response):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
    print(doc.metadata)
    
# combining the page_content from each doc to one
from utils import clean_text
doc_content = []
for doc in response:
    doc_content.append(clean_text(doc.page_content))
print(doc_content)


# similarity search with score
response = vector_store_mongodb.similarity_search_with_score(
    query = "deep learning",
    k = 2
)
print(response)
doc, score = response[0]
print(score)
for i, doc_tuple in enumerate(response):
    print(f"\n--- Result {i+1} ---")
    doc, score = doc_tuple
    print(f"Score = {score}")
    print(doc.page_content)


# return doc based on similarity to an embedding query
response = vector_store_mongodb.similarity_search_by_vector(
    embedding = embedding.embed_query("What is neural network?"),
    k = 2
)
print(response)



# Retriever
# retrivers are not limited to vector stores. They can also pull data from other sources like external APIs, databases, or other systems.
from langchain_core.runnables import chain
from langchain_core.documents import Document
from typing import List

@chain
def retriever(query: str) -> List[Document]:
    return vector_store_mongodb.similarity_search(query, k = 1)

# batch will run in paraller
retriever.batch(
    [
        "What is sigmoid function?",
        "what is ReLu"
    ]
)

# another method
retriever_ = vector_store_mongodb.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k': 2}
)
retriever_.batch(
    [
        "where is sigmoid function is used?",
        "where is ReLu function is used"
    ]
)



# Model
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash"
)



# Prompt
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template(
    '''
    ## QUESTION:
    {question}
    
    ## CONTENT:
    {doc_content}
    
    ## INSTRUCTION:
    Based on the content provided in 'CONTENT' section. Answer the question mentioned in 'QUESTION' section.
    Your answer should be based on the content provided in 'CONTENT' section only.
    If you are not able to get the answer then return the answer as 'No information in the given documents.'
    Do not return preamble in answer.
    
    ## AMSWER (NO PREAMBLE):
    '''
)


# chain
chain = prompt | llm

# defining question
question = "What is mean by Neural Network"

# running the chain
response = chain.invoke(
    input = {
        'question': question,
        'doc_content': doc_content
    }
)
print(response.content)


from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path = "../data/Deep_Learning_A_Visual_Approach.pdf"
)

docs = loader.load()

type(docs)

docs[45]

print(docs[45].page_content)
docs[45].metadata


# lazy_load()
documents_iterator = loader.lazy_load()

type(documents_iterator)

# documents_iterator[45]

# print(documents_iterator.page_content)
# documents_iterator[45].metadata

# Recreate the iterator to ensure it is not exhausted #***************************
documents_iterator = loader.lazy_load()

print(f"documents_iterator type: {type(documents_iterator)}")
for doc in documents_iterator:
    if doc.metadata['page'] == 45:
        print(f"doc type: {type(doc)}")
        print(f"Page no.: {doc.metadata['page']}")
        print(f"{doc.page_content}")
        break
    

# alazy_load()
async def load_document():
    documents_iterator = loader.alazy_load()
    print("documents_iterator type: ", type(documents_iterator))
    async for doc in documents_iterator:
        if doc.metadata['page'] == 45:
            print(f"doc type: {type(doc)}")
            print(f"Page no.: {doc.metadata['page']}")
            print(f"{doc.page_content}")
            break
await load_document()


# load and process pdf pages asynchronously
pages = []
async for page in loader.alazy_load():
    pages.append(page)
    
print(pages[45].metadata)
print(f"page no.: {pages[45].metadata['page']}")
print(pages[45].page_content)



# ************************
#* Vector Search over PDF
# ************************
# craeting vector store
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-mpnet-base-v2"
)

vector_store = InMemoryVectorStore.from_documents(
    pages, embedding
)

# saving the vectoe store
import pickle
with open("../data/vector_store.pkl", 'wb') as f:
    pickle.dump(vector_store, f)

question = "What is meant by Decision Tree?"

docs = vector_store.similarity_search(
    query = question,
    k = 3
)

for doc in docs:
    print(f"Page no.: {doc.metadata['page']}")
    print(doc.page_content)
    

# build the data to pass to the llm
pages_data = []
for doc in docs:
    dict_ = {
        "doc_title": doc.metadata['title'].strip(),
        "page_no": doc.metadata['page_label'].strip(),
        "page_content": doc.page_content.strip()
    }
    pages_data.append(dict_)
   
 
# building LLM model
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path = "../.env")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model = "gemma2-9b-it",
    temperature = 0.8,
    api_key = groq_api_key
)


# make the prompt
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables = ['question', 'pages_data'],
    template = 
    '''
    ## Question:
    {question}
    
    ## Page Data:
    {pages_data}
    
    ## Instruction on what to do:
    In the 'Page data' session I had provided the list of dictonaries.
    Each dictonary have following keys:
        doc_title: title of the document
        page_no.: page number in the document mentioned in doc_title
        page_content: content in the page
    Your job is to analyze the page_content and provide the answer to the question mentioned in the 'Question' session.
    Also mentioned want document and page number you had refered.
    Do not provide preamble.
    
    ## Answer (No Preamble):
    '''
)


# make the chain
from langchain.chains import LLMChain

chain = LLMChain(
    llm = llm,
    prompt = prompt
)

answer = chain.run({'question': question, 'pages_data': pages_data})
print(answer)
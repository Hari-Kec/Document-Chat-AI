from langchain_community.vectorstores import Chroma
from document_processing import get_embeddings
from config import LOCAL_VECTOR_STORE_DIR

def create_vectorstore(documents):
    embeddings = get_embeddings()
    persist_directory = LOCAL_VECTOR_STORE_DIR / "default_collection"
    if persist_directory.exists():
        vectorstore = Chroma(persist_directory=str(persist_directory),
                             embedding_function=embeddings,
                             collection_name="default_collection")
        vectorstore.add_documents(documents)  
    else:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_directory),
            collection_name="default_collection",
        )
    
    vectorstore.persist()
    return vectorstore

def load_vectorstore():
    embeddings = get_embeddings()
    persist_directory = LOCAL_VECTOR_STORE_DIR / "default_collection"
    return Chroma(persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name="default_collection",
    ) 
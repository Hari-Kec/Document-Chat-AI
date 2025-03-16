from langchain.vectorstores.base import VectorStoreRetriever
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer

def create_retriever(vectorstore):
    retriever = VectorStoreRetriever(vectorstore=vectorstore, search_type="similarity", search_kwargs={"k": 5})
    return retriever

def hybrid_search(query, documents, vector_retriever, top_k=5):
    
    tokenized_docs = [doc.page_content.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.split())
    bm25_results = [documents[i] for i in sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)[:top_k]]

    
    vector_results = vector_retriever.get_relevant_documents(query)

    
    combined_results = bm25_results + vector_results
    
    unique_results = []
    seen_content = set()
    for doc in combined_results:
        if doc.page_content not in seen_content:
            unique_results.append(doc)
            seen_content.add(doc.page_content)
    
    return unique_results[:top_k]
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from memory import create_memory
from config import OPENAI_API_KEY, OPENAI_MODEL

def create_conversational_chain(retriever):
    """
    Create a conversational chain with memory and context retrieval.
    
    Args:
        retriever: The retriever object for document retrieval.
    
    Returns:
        chain: The conversational chain.
        memory: The memory object for storing chat history.
    """
    
    memory = create_memory()

    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an intelligent assistant that provides accurate answers based on the given context.\n"
         "If the context does not contain relevant information, say 'I don't know' instead of making up an answer.\n"
         "Ensure responses are concise, well-structured, and relevant to the user's question.\n\n"
         "Context:\n{context}"),
        *memory.load_memory_variables({})["chat_history"],  
        ("human", "{question}"),
    ])

    
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0.7)

    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain, memory

def self_correct_response(response, retriever, query):
    relevant_docs = retriever.get_relevant_documents(query)
    if not relevant_docs:
        return "I don't know."
    return response
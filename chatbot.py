import streamlit as st
from document_processing import load_documents, split_documents
from vectorstore import create_vectorstore
from retrieval import create_retriever, hybrid_search
from chain import create_conversational_chain, self_correct_response
from memory import create_memory, handle_ambiguous_query
import logging
import datetime
import os
from config import LOG_DIR, OPENAI_MODEL

def setup_logging():
    """Set up logging for the chatbot."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y%m%d") + ".log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

def log_query(model, response, prompt_tokens):
    """
    Log the query details, including token usage and cost.
    
    Args:
        model (str): The model used for the response.
        response (AIMessage): The response object from the LLM.
        prompt_tokens (int): The number of tokens used in the prompt.
    """
    input_cost_per_million = 1.10  
    output_cost_per_million = 4.40  
    
    
    response_content = response.content if hasattr(response, "content") else str(response)
    
    
    completion_tokens = len(response_content.split())
    total_tokens = prompt_tokens + completion_tokens
    input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million
    output_cost = (completion_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost
    
    logging.info(
        f"Model: {model}, Completion Tokens: {completion_tokens}, "
        f"Prompt Tokens: {prompt_tokens}, Total Tokens: {total_tokens}, "
        f"Cost: ${total_cost:.6f}"
    )

def chatbot():
    """Main function to run the chatbot."""
    st.title("MAWAD ONLINE RAG CHATBOT")
    setup_logging()

    
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        
        documents = load_documents(uploaded_files)
        chunks = split_documents(documents)
        vectorstore = create_vectorstore(chunks)
        retriever = create_retriever(vectorstore)
        chain, memory = create_conversational_chain(retriever)

        st.success("Vectorstore created successfully!")

        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])  

        
        if prompt := st.chat_input("Ask a question"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)  

            
            if handle_ambiguous_query(prompt, memory):
                chat_history = memory.load_memory_variables({})["chat_history"]
                if chat_history:
                    last_context = chat_history[-1].content
                    clarification = f"Are you referring to {last_context}?"
                else:
                    clarification = "Can you clarify what you're referring to?"
                
                st.session_state.messages.append({"role": "assistant", "content": clarification})
                with st.chat_message("assistant"):
                    st.markdown(clarification)
            else:
                
                relevant_docs = hybrid_search(prompt, chunks, retriever)
                
                
                response = chain.invoke(prompt)
                
                
                response_content = response.content if hasattr(response, "content") else str(response)
                
                
                corrected_response = self_correct_response(response_content, retriever, prompt)

                
                st.session_state.messages.append({"role": "assistant", "content": corrected_response})
                with st.chat_message("assistant"):
                    st.markdown(corrected_response)  

                
                log_query(
                    model=OPENAI_MODEL,
                    response=response,  
                    prompt_tokens=response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
                )

if __name__ == "__main__":
    chatbot()
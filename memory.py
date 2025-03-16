from langchain.memory import ConversationBufferMemory

def create_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",  
        output_key="answer"     
    )

def handle_ambiguous_query(query, memory):
    
    if "it" in query.lower() or "this" in query.lower() or "that" in query.lower():
        
        chat_history = memory.load_memory_variables({})["chat_history"]
        if chat_history:
            
            last_context = chat_history[-1].content
            memory.save_context({"input": query}, {"output": f"Are you referring to {last_context}?"})
            return True
    return False
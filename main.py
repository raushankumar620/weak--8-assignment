from rag_chatbot import RAGChatbot
import os

def main():
    print("🤖 RAG Q&A Chatbot")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = RAGChatbot(use_simple_llm=True)
    
    # Documents folder
    docs_folder = r"c:\Users\kumar\OneDrive\Desktop\week-8\documents"
    
    # Check if vector store exists
    if os.path.exists("vector_store.faiss"):
        print("Loading existing vector store...")
        chatbot.load_vector_store("vector_store")
    else:
        print(f"Loading documents from: {docs_folder}")
        chatbot.load_documents(docs_folder)
        if chatbot.is_initialized:
            chatbot.save_vector_store("vector_store")
    
    if not chatbot.is_initialized:
        print("Failed to initialize chatbot. Exiting...")
        return
    
    print("\nChatbot ready! Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        # Get user question
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        # Get answer
        print("\nThinking...")
        answer = chatbot.ask_question(question)
        print(f"\nAnswer: {answer}")
        
        # Show context option
        show_context = input("\nShow retrieved context? (y/n): ").lower() == 'y'
        if show_context:
            contexts = chatbot.get_relevant_contexts(question)
            print("\nRetrieved Context:")
            for i, (context, score) in enumerate(contexts, 1):
                print(f"\nContext {i} (Score: {score:.3f}):")
                print(context[:300] + "..." if len(context) > 300 else context)

if __name__ == "__main__":
    main()

import streamlit as st
import os
try:
    from rag_chatbot import RAGChatbot
except ImportError as e:
    st.error(f"Error importing RAGChatbot: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="RAG Q&A Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = RAGChatbot(use_simple_llm=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main UI
st.title("🤖 RAG Q&A Chatbot")
st.markdown("Upload documents and ask questions based on their content!")

# Sidebar for document management
with st.sidebar:
    st.header("📁 Document Management")
    
    # Documents folder path
    docs_folder = st.text_input(
        "Documents Folder Path:",
        value=r"c:\Users\kumar\OneDrive\Desktop\week-8\documents"
    )
    
    # Create documents folder if it doesn't exist
    if st.button("Create Documents Folder"):
        try:
            os.makedirs(docs_folder, exist_ok=True)
            st.success(f"Folder created: {docs_folder}")
            st.info("Please add some .txt or .pdf files to this folder")
        except Exception as e:
            st.error(f"Error creating folder: {e}")
    
    # File upload section
    st.subheader("📤 Upload Files")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['txt', 'pdf'],
        accept_multiple_files=True,
        help="Upload .txt or .pdf files to add to your document collection"
    )
    
    if uploaded_files and st.button("Save Uploaded Files"):
        if not os.path.exists(docs_folder):
            os.makedirs(docs_folder, exist_ok=True)
        
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(docs_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(uploaded_file.name)
        
        st.success(f"Saved {len(saved_files)} files: {', '.join(saved_files)}")
    
    # Check folder status
    if os.path.exists(docs_folder):
        files = [f for f in os.listdir(docs_folder) if f.endswith(('.txt', '.pdf'))]
        st.info(f"Folder exists with {len(files)} document files")
        
        if files:
            st.write("📄 Documents found:")
            for file in files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    file_path = os.path.join(docs_folder, file)
                    file_size = os.path.getsize(file_path)
                    st.write(f"• {file} ({file_size} bytes)")
                with col2:
                    if st.button("🗑️", key=f"delete_{file}", help=f"Delete {file}"):
                        try:
                            os.remove(file_path)
                            st.success(f"Deleted {file}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting {file}: {e}")
        else:
            st.warning("No .txt or .pdf files found in folder")
    else:
        st.warning("Folder does not exist")
    
    # Load documents button
    if st.button("Load Documents"):
        if not os.path.exists(docs_folder):
            st.error("Documents folder does not exist. Create it first.")
        else:
            files = [f for f in os.listdir(docs_folder) if f.endswith(('.txt', '.pdf'))]
            if not files:
                st.error("No .txt or .pdf files found in the folder.")
            else:
                with st.spinner("Processing documents..."):
                    success = st.session_state.chatbot.load_documents(docs_folder)
                    if success and st.session_state.chatbot.is_initialized:
                        st.success(f"Successfully loaded {len(files)} documents!")
                        # Save vector store
                        st.session_state.chatbot.save_vector_store("vector_store")
                    else:
                        st.error("Failed to load documents. Check the folder path and file formats.")
    
    # Load existing vector store
    if st.button("Load Existing Vector Store"):
        with st.spinner("Loading vector store..."):
            st.session_state.chatbot.load_vector_store("vector_store")
            if st.session_state.chatbot.is_initialized:
                st.success("Vector store loaded!")
            else:
                st.error("Failed to load vector store.")
    
    # Status
    if st.session_state.chatbot.is_initialized:
        st.success("✅ Chatbot Ready")
    else:
        st.warning("⚠️ Load documents first")

# Main chat interface
st.header("💬 Chat Interface")

# Display chat history
for i, (question, answer) in enumerate(st.session_state.chat_history):
    with st.expander(f"Q{i+1}: {question[:50]}...", expanded=False):
        st.write(f"**Question:** {question}")
        st.write(f"**Answer:** {answer}")

# Question input
question = st.text_input("Ask a question about your documents:")

# Ask button
col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button("Ask Question", type="primary")

with col2:
    show_context = st.checkbox("Show retrieved context")

# Process question
if ask_button and question:
    if st.session_state.chatbot.is_initialized:
        with st.spinner("Generating response..."):
            # Get answer
            answer = st.session_state.chatbot.ask_question(question)
            
            # Add to chat history
            st.session_state.chat_history.append((question, answer))
            
            # Display current Q&A
            st.subheader("Current Response:")
            st.write(f"**Question:** {question}")
            st.write(f"**Answer:** {answer}")
            
            # Show context if requested
            if show_context:
                st.subheader("Retrieved Context:")
                contexts = st.session_state.chatbot.get_relevant_contexts(question)
                for i, (context, score) in enumerate(contexts):
                    with st.expander(f"Context {i+1} (Score: {score:.3f})"):
                        st.write(context)
    else:
        st.error("Please load documents first!")

# Clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

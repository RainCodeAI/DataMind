# the_analyst/app.py

import streamlit as st
import os

from agent_handler import AnalystAgent

# --- Page Configuration ---
st.set_page_config(
    page_title="The Analyst",
    page_icon="📊",
    layout="wide"
)

# --- App State Initialization ---
if "agent" not in st.session_state:
    st.session_state.agent = AnalystAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Rendering ---
st.title("📊 The Analyst")
st.caption("Your personal data analyst. Upload a CSV and start asking questions.")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Data Source")
    st.markdown("""
    Upload a CSV file and The Analyst will help you understand it. 
    You can ask questions about statistics, trends, or specific data points.
    """) # NEW: Added more descriptive text
    uploaded_file = st.file_uploader("Upload your CSV file here:", type=["csv"])

    if uploaded_file is not None:
        # To work with the LangChain agent, we need to save the file temporarily
        temp_dir = "temp_data"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.temp_file_path = temp_file_path
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")


# --- Main Chat Interface ---
# NEW: Display a welcome message if the chat is empty
if not st.session_state.messages:
    st.info("Upload a CSV file using the sidebar to get started!")

# Display chat messages from history
for message in st.session_state.messages:
    # NEW: Added a custom avatar for the assistant
    avatar = "📊" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user", avatar="👤"): # NEW: Added user avatar
        st.markdown(prompt)

    # Check if a file has been uploaded
    if "temp_file_path" in st.session_state:
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="📊"): # NEW: Added assistant avatar
            with st.spinner("Analyzing..."):
                response = st.session_state.agent.get_response(
                    csv_file_path=st.session_state.temp_file_path,
                    user_query=prompt
                )
                st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # This warning will now appear in the chat flow if a file isn't uploaded first
        with st.chat_message("assistant", avatar="📊"):
            st.warning("Please upload a CSV file first to ask questions about it.", icon="⚠️")
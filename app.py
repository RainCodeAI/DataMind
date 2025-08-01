# app.py

import streamlit as st
import os
import tempfile

from agent_handler import AnalystAgent

# --- Page Configuration ---
st.set_page_config(
    page_title="The Analyst",
    page_icon="📊",
    layout="wide"
)

# --- App State Initialization ---
if "agent" not in st.session_state:
    try:
        st.session_state.agent = AnalystAgent()
        st.session_state.demo_mode = False
    except Exception as e:
        st.session_state.agent = None
        st.session_state.demo_mode = True
        st.warning("⚠️ Demo Mode: OpenAI API key not configured. Upload files to see the interface, but analysis features require an API key.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "temp_file_path" not in st.session_state:
    st.session_state.temp_file_path = None

# --- UI Rendering ---
st.title("📊 The Analyst")
if st.session_state.get("demo_mode", False):
    st.caption("Your personal data analyst. Upload a CSV and start asking questions. | 🔄 **Demo Mode** - Configure API key for full analysis")
else:
    st.caption("Your personal data analyst. Upload a CSV and start asking questions.")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Data Source")
    st.markdown("""
    Upload a CSV file and The Analyst will help you understand it. 
    You can ask questions about statistics, trends, or specific data points.
    
    **Example questions:**
    - "What is the average value of column X?"
    - "Show me the top 10 records by revenue"
    - "How many unique categories are there?"
    - "What are the trends over time?"
    """)
    
    uploaded_file = st.file_uploader("Upload your CSV file here:", type=["csv"])

    if uploaded_file is not None:
        try:
            # Create a temporary file to save the uploaded CSV
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_file_path = tmp_file.name
            
            st.session_state.temp_file_path = temp_file_path
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            
            # Display basic file info
            try:
                import pandas as pd
                df = pd.read_csv(temp_file_path)
                st.info(f"**File Info:**\n- Rows: {len(df)}\n- Columns: {len(df.columns)}")
                
                with st.expander("Preview data"):
                    st.dataframe(df.head(), use_container_width=True)
                    
            except Exception as e:
                st.warning(f"Could not preview file: {e}")
                
        except Exception as e:
            st.error(f"Error uploading file: {e}")

# --- Main Chat Interface ---
# Display a welcome message if the chat is empty
if not st.session_state.messages:
    if st.session_state.get("demo_mode", False):
        st.info("👋 Welcome to The Analyst! Currently in **Demo Mode** - upload a CSV file to see the interface in action. Configure your OpenAI API key for full data analysis capabilities.")
        
        # Show example questions in demo mode
        with st.expander("✨ Example questions you can ask (requires API key for actual analysis)"):
            st.markdown("""
            - "What is the average value in the revenue column?"
            - "Show me the top 10 rows by sales amount"
            - "How many unique products are there?"
            - "What are the trends in this data over time?"
            - "Calculate the correlation between price and quantity"
            - "Which category has the highest average rating?"
            - "How many records have missing values?"
            """)
    else:
        st.info("👋 Welcome to The Analyst! Upload a CSV file using the sidebar to get started with data analysis.")

# Display chat messages from history
for message in st.session_state.messages:
    avatar = "📊" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Check if a file has been uploaded
    if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="📊"):
            if st.session_state.demo_mode:
                # Demo mode response
                demo_response = f"""
                **Demo Mode Response** 
                
                I can see you've uploaded a CSV file and asked: "{prompt}"
                
                In the full version with an API key, I would analyze your data and provide insights like:
                - Statistical summaries and calculations
                - Data filtering and trends analysis
                - Specific answers based on your CSV content
                - Charts and visualizations descriptions
                
                To enable full analysis features, please configure your OpenAI API key.
                """
                st.markdown(demo_response)
                st.session_state.messages.append({"role": "assistant", "content": demo_response})
            else:
                # Full mode with actual analysis
                if st.session_state.agent:
                    with st.spinner("Analyzing your data..."):
                        try:
                            response = st.session_state.agent.get_response(
                                csv_file_path=st.session_state.temp_file_path,
                                user_query=prompt
                            )
                            st.markdown(response)
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_message = f"I encountered an error while analyzing your data: {str(e)}"
                            st.error(error_message)
                            st.session_state.messages.append({"role": "assistant", "content": error_message})
                else:
                    error_message = "Agent not properly initialized. Please check your API key configuration."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        # Warning if no file is uploaded
        warning_message = "⚠️ Please upload a CSV file first to ask questions about it."
        with st.chat_message("assistant", avatar="📊"):
            st.warning(warning_message)
        st.session_state.messages.append({"role": "assistant", "content": warning_message})

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 20px;'>
        Powered by <strong>RainCode AI</strong> • 
        Built with <strong>OpenAI GPT</strong>, <strong>LangChain</strong> & <strong>Streamlit</strong>
    </div>
    """, 
    unsafe_allow_html=True
)

# Cleanup temporary files on app restart
def cleanup_temp_files():
    if "temp_file_path" in st.session_state and st.session_state.temp_file_path:
        try:
            if os.path.exists(st.session_state.temp_file_path):
                os.unlink(st.session_state.temp_file_path)
        except Exception:
            pass  # Ignore cleanup errors

# Register cleanup function
import atexit
atexit.register(cleanup_temp_files)

# app.py

import streamlit as st
import os
import tempfile
import pandas as pd

from agent_handler import AnalystAgent
from visualization_engine import VisualizationEngine
from data_profiler import DataProfiler
from multi_file_manager import MultiFileManager
from export_manager import ExportManager
from advanced_analytics import AdvancedAnalytics
from collaboration_manager import CollaborationManager
from ai_insights import AIInsights

# --- Page Configuration ---
st.set_page_config(
    page_title="The Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
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

# Initialize advanced components
if "viz_engine" not in st.session_state:
    st.session_state.viz_engine = VisualizationEngine()

if "data_profiler" not in st.session_state:
    st.session_state.data_profiler = DataProfiler()

if "file_manager" not in st.session_state:
    st.session_state.file_manager = MultiFileManager()

if "export_manager" not in st.session_state:
    st.session_state.export_manager = ExportManager()

if "advanced_analytics" not in st.session_state:
    st.session_state.advanced_analytics = AdvancedAnalytics()

if "collaboration_manager" not in st.session_state:
    st.session_state.collaboration_manager = CollaborationManager()

if "ai_insights" not in st.session_state:
    st.session_state.ai_insights = AIInsights()

if "current_dataframe" not in st.session_state:
    st.session_state.current_dataframe = None

# --- UI Rendering ---
st.title("📊 The Analyst")
if st.session_state.get("demo_mode", False):
    st.caption("Your personal data analyst. Upload a CSV and start asking questions. | 🔄 **Demo Mode** - Configure API key for full analysis")
else:
    st.caption("Your personal data analyst. Upload a CSV and start asking questions.")

# --- Sidebar Navigation ---
with st.sidebar:
    st.header("🎯 Navigation")
    
    # Main navigation
    page = st.selectbox(
        "Choose your workspace:",
        ["💬 Chat Analysis", "📁 File Manager", "📊 Data Profiling", 
         "📈 Visualizations", "🧠 Advanced Analytics", "📤 Export & Reports", 
         "👥 Collaboration", "🤖 AI Insights"]
    )
    
    st.divider()
    
    # File Upload Section
    st.header("📁 Data Sources")
    st.markdown("Upload CSV files for analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"], key="main_uploader")

    if uploaded_file is not None:
        try:
            # Add file to multi-file manager
            file_id = st.session_state.file_manager.add_file(uploaded_file)
            
            if file_id:
                # Set as current file for analysis
                df = st.session_state.file_manager.get_file_dataframe(file_id)
                if df is not None:
                    st.session_state.current_dataframe = df
                    st.session_state.temp_file_path = st.session_state.file_manager.get_all_files()[file_id]['path']
                    
                    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                    
                    # Display basic file info
                    st.info(f"**File Info:**\n- Rows: {len(df):,}\n- Columns: {len(df.columns)}")
                    
                    # Auto-generate insights
                    with st.spinner("Generating AI insights..."):
                        insights = st.session_state.ai_insights.analyze_and_generate_insights(df, uploaded_file.name)
                        if insights:
                            st.session_state.ai_insights.generate_insight_notification(insights[0])
                    
                    with st.expander("📋 Quick Preview"):
                        st.dataframe(df.head(), use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error uploading file: {e}")
    
    # Quick stats for current data
    if st.session_state.current_dataframe is not None:
        st.divider()
        st.header("📊 Current Data")
        df = st.session_state.current_dataframe
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        
        # Data quality score
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        quality_score = max(0, 100 - missing_percentage)
        st.metric("Quality Score", f"{quality_score:.1f}%")
        
        # Smart suggestions
        if st.button("💡 Get Smart Suggestions"):
            recent_queries = [msg['content'] for msg in st.session_state.messages[-5:] if msg['role'] == 'user']
            suggestions = st.session_state.ai_insights.suggest_follow_up_questions(df, recent_queries)
            
            if suggestions:
                st.write("**Suggested Questions:**")
                for suggestion in suggestions:
                    if st.button(f"❓ {suggestion}", key=f"suggestion_{hash(suggestion)}"):
                        # Add suggestion to chat
                        st.session_state.messages.append({"role": "user", "content": suggestion})
                        st.rerun()

# --- Main Content Area ---
if page == "💬 Chat Analysis":
    # Main Chat Interface
    st.header("💬 Chat Analysis")
    
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
                                
                                # Generate visualizations if appropriate
                                if st.session_state.current_dataframe is not None:
                                    chart_suggestions = st.session_state.viz_engine.suggest_charts_for_query(
                                        prompt, st.session_state.current_dataframe
                                    )
                                    
                                    if chart_suggestions:
                                        st.write("**📈 Suggested Visualizations:**")
                                        for suggestion in chart_suggestions[:2]:
                                            if st.button(f"📊 Create {suggestion['title']}", key=f"chart_{hash(suggestion['title'])}"):
                                                # Create and display chart based on suggestion
                                                st.info("Chart generation feature coming soon!")
                                
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

elif page == "📁 File Manager":
    st.session_state.file_manager.render_file_manager_ui()

elif page == "📊 Data Profiling":
    st.header("📊 Data Profiling Dashboard")
    if st.session_state.current_dataframe is not None:
        profile = st.session_state.data_profiler.generate_data_profile(
            st.session_state.current_dataframe, "current_dataset"
        )
        st.session_state.data_profiler.render_profile_dashboard(profile)
    else:
        st.info("Please upload a CSV file to see data profiling insights.")

elif page == "📈 Visualizations":
    st.header("📈 Smart Visualizations")
    if st.session_state.current_dataframe is not None:
        df = st.session_state.current_dataframe
        
        # Analyze data for chart suggestions
        analysis = st.session_state.viz_engine.analyze_data_for_charts(df)
        
        st.write("**📊 Suggested Charts:**")
        for suggestion in analysis['suggested_charts']:
            with st.expander(f"📈 {suggestion['title']}"):
                st.write(suggestion['title'])
                st.info("Interactive chart creation coming soon!")
                
                # Show data preview for the chart
                if suggestion['type'] == 'scatter' and len(analysis['numeric_columns']) >= 2:
                    preview_df = df[[suggestion['x'], suggestion['y']]].head()
                    st.dataframe(preview_df)
        
        # Manual chart creation
        st.write("**🎨 Create Custom Chart:**")
        chart_type = st.selectbox("Chart Type:", ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram"])
        
        if chart_type and analysis['numeric_columns']:
            x_col = st.selectbox("X-axis:", df.columns.tolist())
            y_col = st.selectbox("Y-axis:", analysis['numeric_columns'])
            
            if st.button("Create Chart"):
                try:
                    if chart_type == "Line Chart":
                        st.line_chart(df.set_index(x_col)[y_col])
                    elif chart_type == "Bar Chart":
                        chart_data = df.groupby(x_col)[y_col].mean()
                        st.bar_chart(chart_data)
                    elif chart_type == "Scatter Plot":
                        st.scatter_chart(df, x=x_col, y=y_col)
                    elif chart_type == "Histogram":
                        st.bar_chart(df[y_col].value_counts())
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")
    else:
        st.info("Please upload a CSV file to create visualizations.")

elif page == "🧠 Advanced Analytics":
    if st.session_state.current_dataframe is not None:
        st.session_state.advanced_analytics.render_analytics_dashboard(st.session_state.current_dataframe)
    else:
        st.info("Please upload a CSV file to use advanced analytics features.")

elif page == "📤 Export & Reports":
    st.session_state.export_manager.render_export_interface(
        st.session_state.messages,
        {"dataframe": st.session_state.current_dataframe, "name": "current_dataset"},
        {"conversation": st.session_state.messages}
    )

elif page == "👥 Collaboration":
    st.session_state.collaboration_manager.render_collaboration_interface()

elif page == "🤖 AI Insights":
    if st.session_state.current_dataframe is not None:
        st.session_state.ai_insights.render_insights_dashboard(st.session_state.current_dataframe)
    else:
        st.session_state.ai_insights.render_insights_dashboard(None)

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

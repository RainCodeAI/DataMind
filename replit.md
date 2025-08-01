# Overview

The Analyst is an AI-powered data analysis application that enables users to upload CSV files and ask natural-language questions about their data without requiring any coding skills. The application leverages OpenAI's GPT-4o model combined with LangChain's CSV Agent to provide conversational data exploration and statistical analysis capabilities. Users can upload any CSV file and interact with their data through an intuitive chat interface, asking questions like "What is the average revenue by year?" or "Which products had the highest ratings?" and receive intelligent, context-aware responses.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application uses **Streamlit** as the web framework, providing a clean and interactive user interface. The main components include:
- **Main chat interface**: Conversational area for asking questions and displaying responses
- **Sidebar file uploader**: Dedicated area for CSV file uploads with user guidance
- **Session state management**: Maintains conversation history and uploaded file references across user interactions

## Backend Architecture
The core logic is organized into modular components:
- **AnalystAgent class**: Encapsulates the AI agent functionality and manages the LangChain CSV agent lifecycle
- **Configuration module**: Handles environment variable management and API key validation
- **Streamlit app**: Orchestrates the user interface and coordinates between components

## Data Processing Pattern
The application follows a **temporary file storage** pattern where uploaded CSV files are saved to a local temporary directory to enable the LangChain agent to access and analyze the data. This approach allows the AI agent to work with pandas DataFrames and execute Python code for data analysis.

## AI Integration Architecture
The system uses a **chain-of-thought approach** with LangChain's experimental CSV agent:
- **GPT-4o model**: Configured with temperature=0 for deterministic, analytical responses
- **CSV Agent**: Automatically generates and executes Python code to answer user queries
- **Dangerous code execution**: Explicitly enabled to allow the agent to run pandas operations on the data

## Error Handling Strategy
The application implements **graceful degradation** with comprehensive error handling:
- API key validation at startup
- File existence verification before analysis
- User-friendly error messages for configuration issues
- Robust exception handling throughout the processing pipeline

# External Dependencies

## AI Services
- **OpenAI API**: Primary dependency for GPT-4o model access, requiring API key authentication
- **LangChain**: Framework for building the CSV analysis agent and managing AI interactions
- **LangChain Experimental**: Provides the specialized CSV agent toolkit

## Python Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis (used internally by the LangChain agent)
- **Tabulate**: Data formatting and display utilities
- **Python-dotenv**: Environment variable management for secure API key storage

## File System Dependencies
- **Temporary directory creation**: Local file system access for storing uploaded CSV files
- **File I/O operations**: Reading uploaded files and writing temporary copies for agent access

## Environment Configuration
- **OPENAI_API_KEY**: Required environment variable for API authentication
- **.env file support**: Optional configuration file for local development
- **Environment variable validation**: Startup checks to ensure proper configuration
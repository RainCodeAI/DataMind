# 📊 The Analyst

**Your personal data analyst, powered by AI.**

Upload any CSV file and ask natural-language questions about your data — no coding required. The Analyst uses OpenAI's GPT-4o model and LangChain's CSV Agent to interpret, analyze, and answer questions about your data.

## 🔧 Features

- **Natural Language Queries**: Ask questions in plain English about your data
- **CSV File Support**: Upload any structured CSV file for analysis
- **Interactive Chat Interface**: Conversational interface for data exploration
- **Statistical Analysis**: Get insights on averages, trends, distributions, and more
- **Data Filtering**: Ask about specific subsets or conditions in your data
- **Error Handling**: Robust error handling with helpful feedback

## 🧠 How It Works

1. **Upload a CSV**: Use the sidebar to upload your data file
2. **Ask Questions**: Type natural language questions about your data
3. **Get Insights**: Receive AI-powered analysis and answers

### Example Questions You Can Ask:

- "What is the average revenue by year?"
- "Which products had the highest ratings?"
- "How many records have missing values?"
- "Show me the top 10 customers by sales"
- "What are the trends in this data over time?"
- "Calculate the correlation between price and sales"
- "Which category has the most items?"

## 🚀 Running the Application

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. **Clone or download the application files**

2. **Install dependencies**:
   ```bash
   pip install streamlit langchain langchain-openai langchain-experimental python-dotenv pandas tabulate
   ```

3. **Set up your OpenAI API key**:
   
   **Option 1: Environment Variable**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   **Option 2: Create a .env file**
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py --server.port 5000
   ```

5. **Open your browser** and navigate to `http://localhost:5000`

## 📁 File Structure


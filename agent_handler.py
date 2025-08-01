# agent_handler.py

from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from config import get_openai_api_key
import os

class AnalystAgent:
    def __init__(self):
        """
        Initializes the AnalystAgent.
        This includes setting up the LLM we will use for data analysis.
        """
        # Ensure the API key is available
        api_key = get_openai_api_key()
        
        # Initialize the language model we'll use for the agent.
        # temperature=0 makes the model's output deterministic, which is good for data analysis.
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0
        )
        print("AnalystAgent initialized with OpenAI GPT-4o model.")

    def get_response(self, csv_file_path: str, user_query: str) -> str:
        """
        Takes a path to a CSV file and a user query, and returns the agent's response.

        Args:
            csv_file_path (str): The path to the uploaded CSV file.
            user_query (str): The user's question about the data.

        Returns:
            str: The agent's answer.
        """
        print(f"Creating CSV agent for file: {csv_file_path}")
        
        # Validate that the file exists
        if not os.path.exists(csv_file_path):
            return "Error: The uploaded file could not be found. Please try uploading again."
        
        try:
            # Create the CSV Agent
            # This creates an agent specifically designed to work with CSV data.
            # It gives the LLM access to a Python Pandas DataFrame and allows it 
            # to write and execute code to answer questions.
            agent_executor = create_csv_agent(
                llm=self.llm,
                path=csv_file_path,
                verbose=True,  # Set to True to see the agent's thoughts in the terminal
                allow_dangerous_code=True,  # Required for the agent to execute Python code
                agent_type="openai-tools",  # Use the latest agent type
                handle_parsing_errors=True  # Better error handling
            )
            
            # Enhance the user query with context to get better responses
            enhanced_query = f"""
            You are a professional data analyst. Please analyze the CSV data and answer this question: {user_query}
            
            Guidelines for your response:
            - Provide clear, actionable insights
            - Include specific numbers and statistics when relevant
            - If you create calculations, explain your methodology
            - If the question is unclear, ask for clarification
            - Format your response in a clear, professional manner
            """
            
            # Invoke the Agent
            response = agent_executor.invoke({"input": enhanced_query})
            
            # The response is a dictionary, and the actual answer is in the 'output' key.
            answer = response.get("output", "I couldn't find an answer to your question.")
            
            # Clean up the response if it contains unnecessary technical details
            if "Action:" in answer or "Observation:" in answer:
                # Extract just the final answer if the response contains intermediate steps
                lines = answer.split('\n')
                final_answer_lines = []
                capture = False
                
                for line in lines:
                    if "Final Answer:" in line:
                        capture = True
                    if capture:
                        final_answer_lines.append(line.replace("Final Answer:", "").strip())
                
                if final_answer_lines:
                    answer = '\n'.join(final_answer_lines).strip()
            
            return answer

        except Exception as e:
            error_msg = str(e)
            print(f"Error in agent processing: {error_msg}")
            
            # Provide user-friendly error messages
            if "API key" in error_msg.lower():
                return "Error: There's an issue with the OpenAI API key. Please check your configuration."
            elif "file" in error_msg.lower() or "csv" in error_msg.lower():
                return "Error: There was a problem reading your CSV file. Please ensure it's properly formatted."
            elif "token" in error_msg.lower() or "limit" in error_msg.lower():
                return "Error: The file is too large or complex to process. Please try with a smaller dataset."
            else:
                return f"I encountered an error while analyzing your data: {error_msg}"

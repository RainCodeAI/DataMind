# the_analyst/agent_handler.py

# We need to import the new agent creation function and the OpenAI chat model
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from config import get_openai_api_key

class AnalystAgent:
    def __init__(self):
        """
        Initializes the AnalystAgent.
        This now includes setting up the LLM we will use.
        """
        # Ensure the API key is available
        get_openai_api_key()
        
        # Initialize the language model we'll use for the agent.
        # temperature=0 makes the model's output deterministic, which is good for data analysis.
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        print("AnalystAgent initialized with OpenAI model.")

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
        
        # 1. Create the CSV Agent
        # This is the magic function from LangChain. It creates an agent specifically
        # designed to work with CSV data. It gives the LLM access to a Python Pandas
        # DataFrame and allows it to write and execute code to answer questions.
        # NOTE: `langchain_experimental` contains powerful but potentially changing APIs.
        try:
            agent_executor = create_csv_agent(
                llm=self.llm,
                path=csv_file_path,
                verbose=True, # Set to True to see the agent's thoughts in the terminal
                allow_dangerous_code=True # Required for the agent to execute Python code
            )
            
            # 2. Invoke the Agent
            # We run the agent with the user's query. The agent will figure out
            # what code to write and execute to get the answer from the CSV data.
            response = agent_executor.invoke({"input": user_query})
            
            # The response is a dictionary, and the actual answer is in the 'output' key.
            return response.get("output", "I couldn't find an answer.")

        except Exception as e:
            return f"An error occurred while processing the data: {e}"
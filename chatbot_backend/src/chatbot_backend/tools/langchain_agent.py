import os
from dotenv import load_dotenv
from typing import List
import sys
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage

# Centralized logging
from logging_config import get_logger

logger = get_logger('main')
tools_logger = get_logger('tools')
error_logger = get_logger('errors')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from src.chatbot_backend.tools.custom_tool import track_order, get_style_recommendations, get_product_info
from src.chatbot_backend.tools.n8n_tool import send_customer_details

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


def create_agent() -> AgentExecutor:
    """Create and return a LangChain agent executor."""
    tools: List[BaseTool] = [
        track_order, 
        get_style_recommendations, 
        get_product_info, 
        send_customer_details
    ]

    logger.info("Initializing LangChain Agent with Groq LLM...")

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful customer service AI for 'The Luxe' e-commerce store.\n\n"
         
         "For RETURNS/REFUNDS: If customer wants to return something, collect this info step by step:\n"
         "1. Full name\n"
         "2. Email address\n" 
         "3. Phone number\n"
         "4. Location (city, country)\n"
         "5. Product to return\n"
         "Only use send_customer_details tool when you have ALL 5 pieces of info.\n\n"
         
         "For ORDER TRACKING: Ask for order ID, then use track_order tool.\n\n"
         
         "For STYLING: Use get_style_recommendations tool for design questions.\n\n"
         
         "For PRODUCTS: Use get_product_info tool for product details.\n\n"
         
         "Remember: Check conversation history to avoid asking for info already provided. "
         "Be friendly and conversational. Don't mention tool names to users."),
        
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    try:
        agent = create_tool_calling_agent(llm, tools, prompt)
        logger.info("LangChain agent created successfully.")
    except Exception as e:
        error_logger.error(f"Failed to create agent: {e}", exc_info=True)
        raise

    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        max_iterations=3,
        max_execution_time=30,
        return_intermediate_steps=True,
        early_stopping_method="force",
        handle_parsing_errors=True
    )


if __name__ == "__main__":
    agent_executor = create_agent()
    chat_history = []
    
    print("Agent ready. Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        try:
            logger.info(f"User input: {user_input}")
            response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            chat_history.extend([HumanMessage(content=user_input), AIMessage(content=response['output'])])
            chat_history = chat_history[-10:]  # Keep last 10 messages
            logger.info(f"Agent response: {response['output']}")
            print(f"AI: {response['output']}\n")
        except Exception as e:
            error_logger.error(f"Error during agent execution: {e}", exc_info=True)
            print(f"Error: {e}\n")
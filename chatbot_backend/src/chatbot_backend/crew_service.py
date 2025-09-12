import os
import yaml
import re
from typing import Dict, Optional, Tuple, Any
from dotenv import load_dotenv
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai import LLM

from src.chatbot_backend.crew_tools import StyleAdvisorTool, ReturnsTool, VectorDBTool
from src.chatbot_backend.tools.order_api_tool import OrderApiTool

from logging_config import get_logger

logger = get_logger('crew_service')

class CrewAIService:
    """Service class to handle CrewAI functionality without affecting existing tools."""
    
    def __init__(self):
        self.tools_dict = None
        self.gemini_llm = None
        self.agents_config = None
        self.tasks_config = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the CrewAI service components."""
        if self._initialized:
            return True
            
        try:
            logger.info("Initializing CrewAI service...")
            
            # Load environment variables
            load_dotenv()
            os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")
            
            if not os.getenv("GEMINI_API_KEY"):
                logger.error("GOOGLE_API_KEY environment variable not set")
                return False
            
            # Initialize LLM
            self.gemini_llm = LLM(
                model="gemini/gemini-1.5-flash",
                api_key=os.environ["GEMINI_API_KEY"]
            )
            
            # Initialize tools
            self.tools_dict = self._initialize_tools()
            
            # Load configurations
            self.agents_config = self._load_yaml_config("agents.yaml")
            self.tasks_config = self._load_yaml_config("tasks.yaml")
            
            if not self.agents_config or not self.tasks_config:
                logger.error("Failed to load YAML configurations")
                return False
            
            self._initialized = True
            logger.info("CrewAI service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CrewAI service: {e}", exc_info=True)
            return False
    
    def _initialize_tools(self) -> Dict[str, BaseTool]:
        """Initialize all CrewAI tools."""
        tools = {}
        
        tool_configs = [
            ("OrderApiTool", OrderApiTool),  
            ("ReturnsTool", ReturnsTool),
            ("StyleAdvisorTool", StyleAdvisorTool),
            ("ProductInfoTool", VectorDBTool),
        ]
        
        for tool_name, tool_class in tool_configs:
            try:
                tool_instance = tool_class()
                if isinstance(tool_instance, BaseTool):
                    tools[tool_name] = tool_instance
                    logger.info(f"Initialized {tool_name} successfully")
                else:
                    logger.warning(f"{tool_name} is not a valid BaseTool instance")
            except Exception as e:
                logger.error(f"Failed to initialize {tool_name}: {e}")
        
        logger.info(f"Total tools initialized: {len(tools)}")
        return tools
    
    def _load_yaml_config(self, filepath: str) -> Dict[str, Any]:
        """Load configuration from a YAML file."""
        abs_path = os.path.join(os.path.dirname(__file__), "config", filepath)
        try:
            with open(abs_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file '{abs_path}' was not found")
            return {}
        except Exception as e:
            logger.error(f"Error loading YAML config {filepath}: {e}")
            return {}
    
    def _route_query(self, query: str) -> str:
        """Route the user query to determine the appropriate agent."""
        try:
            router_agent = Agent(
                role="Customer Service Router",
                goal="Analyze the user's request and route it.",
                backstory="You are the first point of contact.",
                llm=self.gemini_llm,
                verbose=False
            )
            
            router_task = Task(
                description=f"Analyze the following customer query: '{query}'. Your final answer must be a single word from this list: 'track', 'style', 'return', or 'info'. Do not include any other text or explanation.",
                expected_output="A single word from the list 'track', 'style', 'return', or 'info' that best categorizes the customer's intent.",
                agent=router_agent
            )
            
            router_crew = Crew(
                agents=[router_agent], 
                tasks=[router_task], 
                process=Process.sequential,
                verbose=False
            )
            
            router_result = router_crew.kickoff()
            final_output_text = str(router_result).strip().lower()
            
            # Cleaning up the output to ensure I get just the routing decision
            valid_routes = ['track', 'style', 'return', 'info']
            for route in valid_routes:
                if route in final_output_text:
                    return route
            
            logger.warning(f"Could not determine route from: {final_output_text}")
            return "info"  # Default fallback
            
        except Exception as e:
            logger.error(f"Error in routing: {e}", exc_info=True)
            return "info"  # Default fallback
    
    def _create_agent_and_task(self, agent_key: str, task_key: str, query: str) -> Tuple[Optional[Agent], Optional[Task]]:
        """Create an agent and task based on configuration."""
        agent_config = self.agents_config.get(agent_key)
        task_config = self.tasks_config.get(task_key)

        if not agent_config or not task_config:
            logger.error(f"Configuration not found for {agent_key} or {task_key}")
            return None, None
        
        # Build agent tools
        agent_tools = []
        required_tools = agent_config.get("tools", [])
        
        for tool_name in required_tools:
            if tool_name in self.tools_dict and self.tools_dict[tool_name] is not None:
                agent_tools.append(self.tools_dict[tool_name])
                logger.debug(f"Added tool: {tool_name}")
            else:
                logger.warning(f"Tool {tool_name} not found or None")
        
        try:
            specialized_agent = Agent(
                role=agent_config["role"],
                goal=agent_config["goal"],
                backstory=agent_config["backstory"],
                tools=agent_tools,
                llm=self.gemini_llm,
                verbose=True
            )
            
            specialized_task = Task(
                description=f"{task_config['description']}. Customer query: '{query}'",
                expected_output=task_config["expected_output"],
                agent=specialized_agent
            )
            
            return specialized_agent, specialized_task
            
        except Exception as e:
            logger.error(f"Error creating agent and task: {e}", exc_info=True)
            return None, None
    
    def process_query(self, query: str) -> str:
        """Process a user query using the CrewAI system."""
        if not self._initialized:
            if not self.initialize():
                return "Service initialization failed. Please try again later."
        
        try:
            # Route the query
            route = self._route_query(query)
            logger.info(f"Query routed to: {route}")
            
            route_mapping = {
                "track": ("order_tracking_agent", "track_order_task"),
                "return": ("returns_agent", "return_process_task"),
                "style": ("style_advisor_agent", "style_recommendation_task"),
                "info": ("product_info_agent", "product_info_task")
            }
            
            if route not in route_mapping:
                return "I couldn't understand your request. Please try rephrasing it."
            
            agent_key, task_key = route_mapping[route]
            
            # Create agent and task
            agent, task = self._create_agent_and_task(agent_key, task_key, query)
            
            if not agent or not task:
                return "I encountered an error processing your request. Please try again."
            
            # Execute the crew
            final_crew = Crew(
                agents=[agent], 
                tasks=[task], 
                process=Process.sequential, 
                verbose=False
            )
            
            result = final_crew.kickoff()
            return str(result)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return "I encountered an unexpected error. Please try again later."

# Global service instance
_crew_service = None

def get_crew_service() -> CrewAIService:
    """Get singleton CrewAI service instance."""
    global _crew_service
    if _crew_service is None:
        _crew_service = CrewAIService()
    return _crew_service
import os
import sys
from typing import Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from src.chatbot_backend.tools.n8n_tool import send_customer_details
from src.chatbot_backend.tools.custom_tool import get_vector_client
from services.api_client import EcommerceAPIClient

from logging_config import get_logger

logger = get_logger('crew_tools')

# Initialize API client
api_client = EcommerceAPIClient(base_url="http://127.00.0.1:8001")


class StyleAdvisorTool(BaseTool):
    name: str = "style_advisor"
    description: str = "Provide interior design and styling recommendations based on user preferences, room types, and style preferences."

    def _run(self, query: str) -> str:
        """Get style recommendations based on user query."""
        try:
            if not query or query.strip() == "":
                return "Please provide details about your style preferences or the room you're decorating."
            
            logger.info(f"Getting style recommendations for: {query}")
            
            client = get_vector_client()
            
            # Searching for products in the vector store
            where_filter = {"source": "product"}
            results = client.query(
                query, 
                n_results=5,
                where_filter=where_filter
            )
            
            if not results:
                # Fallback
                results = client.query(query, n_results=5)
            
            if not results:
                return (
                    "I couldn't find specific products matching your style preferences in our catalog. "
                    "Could you try describing your preferences differently, or let me know what specific "
                    "items you're looking for (e.g., sofas, tables, lighting)?"
                )
            
            # Format recommendations
            response = f"Based on your style preferences, here are my recommendations:\n\n"
            
            for i, product in enumerate(results[:5], 1):
                name = product['metadata'].get('name', 'Unknown Product')
                price = product['metadata'].get('price', 'Price not available')
                description = product['metadata'].get('description', product.get('document', ''))
                
                # Truncate description if too long
                if len(description) > 150:
                    description = description[:150] + "..."
                
                response += f"{i}. **{name}** - {price}\n"
                response += f"    {description}\n\n"
            
            response += "Would you like more details about any of these items, or would you prefer recommendations for a different style or room?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in style advisor: {e}", exc_info=True)
            return (
                f"I encountered an issue while searching for recommendations: {str(e)}. "
                "Please try rephrasing your request or contact support if the problem persists."
            )


class ProductInfoTool(BaseTool):
    name: str = "product_info"
    description: str = "Get detailed product information, specifications, and answer frequently asked questions about products."

    def _run(self, query: str) -> str:
        """Get product information based on query."""
        try:
            logger.info(f"Getting product info for: {query}")
            
            client = get_vector_client()
            
            where_filter = None 
            if "faq" in query.lower() or "question" in query.lower():
                where_filter = {"source": "faq"}
            elif "product" in query.lower() or "item" in query.lower():
                where_filter = {"source": "product"}
                
            results = client.query(query, n_results=3, where_filter=where_filter)
            
            if not results:
                return "No relevant product information found. Could you please rephrase your question or be more specific?"
            
            formatted_results = ""
            for i, doc in enumerate(results, 1):
                source = doc['metadata'].get('source', 'unknown')
                
                if source == 'product':
                    name = doc['metadata'].get('name', 'N/A')
                    category = doc['metadata'].get('category', 'N/A')
                    price = doc['metadata'].get('price', 'N/A')
                    content = doc['document']
                    
                    formatted_results += (
                        f"{i}. **{name}**\n"
                        f"    Category: {category}\n"
                        f"    Price: {price}\n"
                        f"    Details: {content}\n\n"
                    )
                elif source == 'faq':
                    content = doc['document']
                    formatted_results += f"{i}. {content}\n\n"
                else:
                    formatted_results += f"{i}. {doc['document']}\n\n"
            
            return formatted_results.strip()
            
        except Exception as e:
            logger.error(f"Error in product info: {e}", exc_info=True)
            return f"An error occurred while retrieving product information: {e}"


class ReturnsTool(BaseTool):
    name: str = "returns_processing"
    description: str = "Handle return requests by collecting customer details and sending them to an external workflow. The input must be a JSON string with 'name', 'phone', 'email', 'product', and 'location'."
    
    def _run(self, query: str) -> str:
        """Process a return request by collecting all necessary information."""
        try:
            logger.info(f"Processing return request with query: {query}")
            
            # Parse the incoming JSON string to extract the arguments
            try:
                data = json.loads(query)
                name = data.get("name")
                phone = data.get("phone")
                email = data.get("email")
                product = data.get("product")
                location = data.get("location")
            except json.JSONDecodeError:
                return "Invalid input format. The tool requires a valid JSON string."
            
            # Check for missing required fields
            if not all([name, phone, email, product, location]):
                return "Missing required fields. Please provide name, phone, email, product, and location."

            # Call the imported function directly with the parsed arguments
            return send_customer_details(
                name=name,
                phone=phone,
                email=email,
                product=product,
                location=location
            )

        except Exception as e:
            logger.error(f"Error in returns processing: {e}", exc_info=True)
            return f"An unexpected error occurred while processing the request: {str(e)}"

class VectorDBTool(ProductInfoTool):
    """Alias for ProductInfoTool to maintain compatibility with existing configurations."""
    name: str = "vector_db_search"
    description: str = "Search the product database and FAQ system for relevant information."
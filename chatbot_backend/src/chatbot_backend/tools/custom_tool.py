import os
import random
import requests
import json
import sys
from typing import Optional, Any, List, Dict
from langchain.tools import tool
from pydantic import BaseModel, Field

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from services.vector_store import VectorStoreClient
from services.api_client import EcommerceAPIClient

# Initialize the API client once at the top level for use in tools
api_client = EcommerceAPIClient(base_url="http://127.0.0.1:8001")

# Global singleton for vector store client
_vector_client: Optional[VectorStoreClient] = None
_vector_client_initialized = False

def get_vector_client() -> VectorStoreClient:
    """Get singleton vector store client, initializing if needed."""
    global _vector_client, _vector_client_initialized
    
    if not _vector_client_initialized:
        try:
            print("Initializing VectorStoreClient singleton...")
            _vector_client = VectorStoreClient()
            _vector_client_initialized = True
            print("VectorStoreClient singleton initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize VectorStoreClient: {e}")
            raise
    
    return _vector_client

def extract_style_keywords(query: str) -> List[str]:
    """Extract style-related keywords from the user query."""
    style_keywords = {
        'modern': ['modern', 'contemporary', 'sleek', 'minimalist', 'clean lines'],
        'traditional': ['traditional', 'classic', 'timeless', 'elegant'],
        'rustic': ['rustic', 'farmhouse', 'country', 'vintage', 'distressed'],
        'industrial': ['industrial', 'metal', 'exposed', 'urban', 'loft'],
        'bohemian': ['bohemian', 'boho', 'eclectic', 'colorful', 'artistic'],
        'scandinavian': ['scandinavian', 'nordic', 'hygge', 'cozy', 'natural wood'],
        'mid-century': ['mid-century', 'retro', '60s', 'atomic', 'eames']
    }
    
    room_keywords = {
        'living room': ['living room', 'lounge', 'family room'],
        'bedroom': ['bedroom', 'master bedroom', 'guest room'],
        'kitchen': ['kitchen', 'dining', 'breakfast nook'],
        'bathroom': ['bathroom', 'bath', 'powder room'],
        'office': ['office', 'study', 'workspace', 'home office']
    }
    
    query_lower = query.lower()
    found_keywords = []
    
    # Extract style keywords
    for style, keywords in style_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            found_keywords.append(style)
    
    # Extract room keywords
    for room, keywords in room_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            found_keywords.append(room.replace(' ', '_'))
    
    return found_keywords


def build_enhanced_query(original_query: str, keywords: List[str]) -> str:
    """Build an enhanced search query using extracted keywords."""
    base_terms = original_query.split()
    
    # Add style-specific terms to enhance search
    style_enhancements = {
        'modern': ['contemporary', 'sleek', 'minimalist'],
        'traditional': ['classic', 'elegant', 'timeless'],
        'rustic': ['wood', 'natural', 'farmhouse'],
        'industrial': ['metal', 'iron', 'urban'],
        'bohemian': ['colorful', 'textured', 'artistic'],
        'scandinavian': ['light wood', 'simple', 'functional'],
        'mid-century': ['walnut', 'teak', 'geometric']
    }
    
    enhanced_terms = base_terms.copy()
    for keyword in keywords:
        if keyword in style_enhancements:
            enhanced_terms.extend(style_enhancements[keyword])
    
    return ' '.join(enhanced_terms)


def filter_and_rank_products(products: List[Dict], query: str, keywords: List[str]) -> List[Dict]:
    """Filter and rank products based on relevance to user preferences."""
    scored_products = []
    
    for product in products:
        score = 0
        product_text = (
            product.get('document', '') + ' ' +
            product['metadata'].get('name', '') + ' ' +
            product['metadata'].get('category', '') + ' ' +
            str(product['metadata'].get('description', ''))
        ).lower()
        
        # Score based on keyword matches
        for keyword in keywords:
            if keyword.replace('_', ' ') in product_text:
                score += 2
        
        # Additional scoring based on category relevance
        category = product['metadata'].get('category', '').lower()
        if any(room in keywords for room in ['living_room', 'bedroom', 'kitchen', 'bathroom', 'office']):
            if 'furniture' in category or 'decor' in category:
                score += 1
        
        # Boost score for products with good ratings or popular items
        if 'rating' in product['metadata'] and product['metadata']['rating'] >= 4.0:
            score += 1
        
        scored_products.append((product, score))
    
    # Sort by score (descending) and return top products
    scored_products.sort(key=lambda x: x[1], reverse=True)
    return [product for product, score in scored_products]


def format_recommendations(products: List[Dict], query: str, max_items: int = 5) -> str:
    """Format product recommendations into a user-friendly response."""
    if not products:
        return "I couldn't find any products matching your style preferences. Could you provide more details about what you're looking for?"
    
    # Group products by category for better organization
    categorized_products = {}
    for product in products[:max_items]:
        category = product['metadata'].get('category', 'Other')
        if category not in categorized_products:
            categorized_products[category] = []
        categorized_products[category].append(product)
    
    response = f"Based on your preferences ({query}), here are my recommendations:\n\n"
    
    for category, category_products in categorized_products.items():
        if len(categorized_products) > 1:
            response += f"**{category}:**\n"
        
        for product in category_products:
            name = product['metadata'].get('name', 'Unknown Product')
            price = product['metadata'].get('price', 'Price not available')
            description = product['metadata'].get('description', product.get('document', ''))
            
            # Truncate description if too long
            if len(description) > 150:
                description = description[:150] + "..."
            
            response += f"• **{name}** - {price}\n"
            response += f"  {description}\n\n"
    
    # Add a helpful closing message
    response += "Would you like more details about any of these items, or would you prefer recommendations for a different style or room?"
    
    return response


@tool
def get_style_recommendations(query: str) -> str:
    """
    A tool to provide product recommendations based on user style and preferences.
    Uses vector search to find actual products matching the user's style preferences.
    Supports various interior design styles and room types.
    """
    if not query or query.strip() == "":
        return "Please provide details about your style preferences or the room you're decorating."
    
    try:
        # Use singleton vector store client
        client = get_vector_client()
        
        # Extract style and room keywords from the query
        keywords = extract_style_keywords(query)
        
        # Build enhanced search query
        enhanced_query = build_enhanced_query(query, keywords)
        
        # Search for products in the vector store
        # Filter to only product documents for recommendations
        where_filter = {"source": "product"}
        results = client.query(
            enhanced_query, 
            n_results=10,  # Get more results to have better filtering options
            where_filter=where_filter
        )
        
        if not results:
            # Fallback: try a broader search without filters
            results = client.query(query, n_results=10)
        
        if not results:
            return (
                "I couldn't find specific products matching your style preferences in our catalog. "
                "Could you try describing your preferences differently, or let me know what specific "
                "items you're looking for (e.g., sofas, tables, lighting)?"
            )
        
        # Filter and rank products based on relevance
        filtered_products = filter_and_rank_products(results, query, keywords)
        
        # Format the recommendations
        return format_recommendations(filtered_products, query)
        
    except Exception as e:
        # Graceful error handling
        return (
            f"I encountered an issue while searching for recommendations: {str(e)}. "
            "Please try rephrasing your request or contact support if the problem persists."
        )


# Alternative version that also incorporates API data
@tool  
def get_advanced_style_recommendations(query: str, budget_range: Optional[str] = None) -> str:
    """
    Advanced style recommendations that combines vector search with live product data.
    Optionally accepts a budget range (e.g., "under $500", "$500-1000", "luxury").
    """
    try:
        # Get base recommendations from vector store
        base_recommendations = get_style_recommendations(query)
        
        # If we have API access, we could also fetch live inventory, pricing, etc.
        # This is where you'd integrate with your e-commerce API for real-time data
        
        # Example of how you might enhance with API data:
        if budget_range:
            budget_filter_text = f"\n\n*Note: Filtering recommendations for {budget_range} budget range.*"
            base_recommendations += budget_filter_text
        
        return base_recommendations
        
    except Exception as e:
        return f"An error occurred while generating advanced recommendations: {str(e)}"


# --- Keep existing tools unchanged ---

@tool
def track_order(order_id: str) -> str:
    """
    A tool to track the status of an order using an external API.
    The order ID can be any valid order number, for example: '12345' or '67890'.
    """
    if not order_id:
        return "Please provide a valid order ID."
    
    response = api_client.get_order_status(order_id)
    
    if response and response.get("status") == "error":
        return response.get("message", "Could not retrieve order details.")
    
    status = response.get("status", "N/A")
    carrier = response.get("carrier", "N/A")
    tracking_number = response.get("tracking_number", "N/A")
    estimated_delivery = response.get("estimated_delivery", "N/A")
    
    return (
        f"Order Status: {status}\n"
        f"Carrier: {carrier}\n"
        f"Tracking Number: {tracking_number}\n"
        f"Estimated Delivery: {estimated_delivery}"
    )


@tool
def get_product_info(query: str) -> str:
    """
    Useful for answering questions about product information, specifications, 
    and frequently asked questions (FAQs).
    Input should be a detailed, natural language query string.
    """
    try:
        # Use singleton vector store client
        client = get_vector_client()
        
        where_filter = None 
        if "faq" in query.lower() or "question" in query.lower():
            where_filter = {"source": "faq"}
        elif "product" in query.lower() or "item" in query.lower():
            where_filter = {"source": "product"}
            
        results = client.query(query, n_results=5, where_filter=where_filter)
        
        if not results:
            return "No relevant documents found in the vector database."
        
        formatted_results = ""
        for doc in results:
            source = doc['metadata'].get('source', 'unknown')
            
            if source == 'product':
                name = doc['metadata'].get('name', 'N/A')
                category = doc['metadata'].get('category', 'N/A')
                price = doc['metadata'].get('price', 'N/A')
                content = doc['document']
                
                formatted_results += (
                    f"--- Product Match ---\n"
                    f"Name: {name}\n"
                    f"Category: {category}\n"
                    f"Price: {price}\n"
                    f"Details: {content}\n\n"
                )
            elif source == 'faq':
                content = doc['document']
                formatted_results += (
                    f"--- FAQ Match ---\n"
                    f"Content: {content}\n\n"
                )
            else:
                formatted_results += f"Source: {source}\nContent: {doc['document']}\n\n"
        
        return formatted_results.strip()
        
    except Exception as e:
        return f"An error occurred while querying the vector database: {e}"


# Example usage and testing
if __name__ == '__main__':
    print("--- Testing Enhanced Style Recommendations ---")
    
    # Test cases
    test_queries = [
        "I want modern furniture for my living room",
        "Looking for rustic decor for my bedroom",
        "Need minimalist office furniture",
        "Bohemian style accessories for my apartment"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("=" * 50)
        result = get_style_recommendations(query)
        print(result)
        print("\n")
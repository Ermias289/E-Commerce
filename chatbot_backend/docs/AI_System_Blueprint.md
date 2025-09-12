# 🤖 Luxe GenAI System Architecture

This blueprint outlines the full AI system architecture for the LUXE e-commerce chatbot, showing the flow of a user's request from the frontend to the final response, integrating all required components.

## 1. High-Level Components

* **FastAPI Chat Service** – entry point for all user queries (`/chat` endpoint).
* **Router** – decides between simple FAQ vs. complex multi-step queries.
* **LangChain Agent** – handles simple tasks (FAQs, product search).
* **CrewAI Orchestrator** – coordinates multi-agent workflows.
* **Specialized Agents** – each agent is focused on a domain (style, orders, returns).
* **MCP Tools** – standardized APIs that connect AI → real data/services.
* **Vector Database (Pinecone, RAG)** – powers semantic search for products & FAQs.

---

## 2. GenAI Flow Diagram

```mermaid
flowchart TB
    U[User Query] --> API[FastAPI /chat Endpoint]

    API --> RT[Router]

    RT -->|Simple FAQ Product| LC[LangChain Agent]
    RT -->|Complex Multi-step| CR[CrewAI Orchestrator]

    LC --> VDB[(Pinecone Vector DB - Product Embeddings)]
    LC --> FAQ[Knowledge Base - Home Decor FAQs]

    CR --> SA[Style Advisor Agent]
    CR --> OTA[Order Tracking Agent]
    CR --> RA[Returns Agent]

    OTA --> ORDAPI[E-commerce Order API]
    OTA --> SHIP[Shipping API]
    RA --> N8N[n8n Workflow - Returns and Emails]
    SA --> VDB
```

---

## 3. CrewAI Agents & Roles

* **Style Advisor Agent**

  * Suggests products that match user's room, mood board, or dimensions.
  * Uses Pinecone vector DB + product metadata.

* **Order Tracking Agent**

  * Fetches order + shipping status.
  * Calls: `get_order_status(order_id)`.

* **Returns Agent**

  * Starts a return process.
  * Calls: `initiate_return(order_id)` via n8n automation.

---

## 4. MCP Tool Contracts

```python
# Product Search
def search_products(query: str) -> List[Dict]

# Order Tracking
def get_order_status(order_id: str) -> Dict

# Add to Cart
def add_to_cart(product_id: str, quantity: int) -> str

# Returns
def initiate_return(order_id: str) -> str
```

---

## 5. Data Flow Examples

1. **FAQ Question**

   * User: "How do I clean velvet?"
   * Router → LangChain Agent → Knowledge Base → Returns FAQ Answer.

2. **Product Search**

   * User: "Show me round wooden coffee tables."
   * LangChain Agent → `search_products()` → Pinecone Vector DB → Suggests items.

3. **Order Tracking**

   * User: "Where's my order #9876?"
   * Router → CrewAI Orchestrator → OrderTrackingAgent → E-commerce API + Shipping API → Returns formatted response.

4. **Return Request**

   * User: "I want to return my chair."
   * Router → CrewAI Orchestrator → ReturnsAgent → `initiate_return()` → n8n triggers → User gets email + return label.

---

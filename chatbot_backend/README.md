# Luxe E-Commerce AI Backend

This repository hosts the backend system for The Luxe e-commerce platform, designed to deliver an intelligent, AI-powered customer support experience. The system integrates multiple cutting-edge technologies: CrewAI orchestrates task-specific agents for complex workflows, LangChain powers conversational AI for FAQs and product recommendations, and a Pinecone vector database enables semantic search across products and knowledge bases.

Key functionalities include:

Automated Customer Support – Handles user queries seamlessly with multi-step reasoning.

Order Tracking – Provides real-time order and shipping status updates.

Returns Management – Guides users through the return process, triggering automated workflows via n8n.

Style Recommendations – Offers personalized product suggestions based on user preferences and room context.

Product Information Retrieval – Delivers detailed product details, specifications, and FAQ responses using retrieval-augmented generation (RAG).

The architecture is modular, scalable, and containerized, allowing for the easy addition of new agents, tools, or data sources, making it a robust foundation for a next-generation luxury e-commerce experience.
---

## Table of Contents

1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Docker Deployment](#docker-deployment)
6. [CrewAI Service](#crewai-service)
7. [LangChain Agent](#langchain-agent)
8. [Tools](#tools)
9. [Vector Database](#vector-database)
10. [Logging](#logging)
11. [Usage Examples](#usage-examples)
12. [Configuration](#configuration)

---

## Features

* **Order Tracking**: Track orders using CrewAI or LangChain agents
* **Returns Processing**: Collect customer details and submit return requests
* **Style Recommendations**: Personalized interior design suggestions
* **Product Information**: Retrieve product details and FAQs
* **Vector Database Search**: High-performance product & FAQ search
* **Centralized Logging**: Full logging for all components
* **Hybrid AI Workflows**: CrewAI + LangChain integration

---

## Technologies Used

* **Python 3.11**
* **CrewAI**: Agent-based automation for customer service tasks
* **LangChain**: Conversational AI agent using Groq LLM
* **Groq LLM** (`llama-3.3-70b-versatile`)
* **Vector Database**: pinecone and cohere for product & FAQ embeddings
* **FastAPI**: API service (via `mock_api.py`)
* **Pydantic**: Data validation for tools
* **YAML**: Configuration files for agents & tasks
* **Docker**: Containerized deployment
* **Logging**: Centralized logging with rotating files (`logging_config.py`)
* **Environment Management**: `python-dotenv`

---

## Project Structure

```
.
├── __pycache__/          # Python cache
├── Dockerfile
├── logging_config.py     # Centralized logging
├── README.md
├── requirements.txt
├── pyproject.toml
├── db.json               # Sample database file
├── data/                 # Optional datasets
├── docs/                 # Documentation (AI blueprint)
│   └── AI_System_Blueprint.md
├── logs/                 # Rotating log files
├── knowledge/            # Product & FAQ JSONs
│   ├── categories.json
│   ├── faqs.json
│   ├── interaction_examples.json
│   ├── niches.json
│   ├── overview_and_strategy.json
│   ├── policies.json
│   ├── product_tips.json
│   ├── promotions.json
│   ├── system_info.json
│   ├── user_experience.json
│   └── users.json
├── services/             # API clients, vector DB utilities
│   ├── __init__.py
│   ├── api_client.py
│   └── vector_store.py
├── src/
│   └── chatbot_backend/
│       ├── __init__.py
│       ├── crew.py
│       ├── crew_service.py     # CrewAI backend service
│       ├── crew_tools.py       # CrewAI tools
│       ├── main.py             # Entry point for the backend
│       ├── mock_api.py
│       ├── config/             # YAML configurations
│       │   ├── agents.yaml
│       │   └── tasks.yaml
│       └── tools/              # Tool implementations
│           ├── __init__.py
│           ├── custom_tool.py
│           ├── n8n_tool.py
│           ├── langchain_agent.py
│           ├── order_api_tool.py
│           └── vector_db_tool.py
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Ermias289/E-Commerce.git
cd E-Commerce

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables in .env
GROQ_API_KEY=<your-groq-api-key>
GOOGLE_API_KEY=<your-google-api-key>
```

---

## Docker Deployment

```bash
# Build Docker image
docker build -t luxe-api .

# Run Docker container
docker run -p 8000:8000 luxe-api

# Access API at http://localhost:8000
```

---

## CrewAI Service

**File:** `src/chatbot_backend/crew_service.py`

* Handles **agent/task orchestration** for customer service.
* Routes queries to agents:

  * `order_tracking_agent` → track orders
  * `returns_agent` → process returns
  * `style_advisor_agent` → provide style recommendations
  * `product_info_agent` → fetch product/FAQ info
* Loads **tools dynamically** (`crew_tools.py`).
* Reads **configuration** from YAML files (`agents.yaml`, `tasks.yaml`).

**Example:**

```python
from src.chatbot_backend.crew_service import get_crew_service

service = get_crew_service()
response = service.process_query("I want to return a sofa I bought last week.")
print(response)
```

---

## LangChain Agent

**File:** `src/chatbot_backend/tools/langchain_agent.py`

* Conversational AI agent using **Groq LLM** (`llama-3.3-70b-versatile`).
* Supports **tools integration**:

  * `track_order`
  * `get_style_recommendations`
  * `get_product_info`
  * `send_customer_details`
* Maintains **chat history** for coherent dialogue.
* Provides **interactive console** for testing.
* Includes **health check** and structured error handling.

**Usage:**

```python
from src.chatbot_backend.tools.langchain_agent import create_agent, health_check

agent_executor = create_agent()
print("Agent ready:", health_check())

response = agent_executor.invoke({
    "input": "Show me sofas for a modern living room",
    "chat_history": []
})
print(response['output'])
```

---

## Tools

| Tool Name                   | Functionality                               |
| --------------------------- | ------------------------------------------- |
| `OrderApiTool`              | Track customer orders                       |
| `ReturnsTool`               | Process return requests                     |
| `StyleAdvisorTool`          | Provide interior design recommendations     |
| `ProductInfoTool`           | Get product information and FAQs            |
| `VectorDBTool`              | Alias for `ProductInfoTool` (vector search) |
| `track_order`               | LangChain tool for order tracking           |
| `get_style_recommendations` | LangChain tool for style suggestions        |
| `get_product_info`          | LangChain tool for product info             |
| `send_customer_details`     | LangChain tool for returns workflow         |

---

## Vector Database

* Stores **product information**, **FAQs**, and **embeddings** for fast retrieval.
* Uses **pinecone, an API based vectordb**.
* Each document has metadata fields like `name`, `price`, `category`, and `source`.
* Tools query the vector DB to provide recommendations and answers.

---

## Logging

Centralized logging is implemented in `logging_config.py`:

* Loggers for **main**, **api**, **tools**, **vector\_db**, and **errors**.
* Rotating files for persistence (`logs/` folder).
* Console logging for development.

---

## Usage Examples

```python
from src.chatbot_backend.crew_service import get_crew_service

service = get_crew_service()

# Example 1: Track order
print(service.process_query("Where is my order #12345?"))

# Example 2: Style advice
print(service.process_query("I want modern style suggestions for my living room"))

# Example 3: Product information
print(service.process_query("Tell me about the new sofa collection"))

# Example 4: Returns
print(service.process_query('{"name":"John Doe","phone":"123456789","email":"john@example.com","product":"Sofa","location":"Nairobi"}'))
```

---

## Configuration

* **Environment variables:** `.env` file for API keys:

  * `GROQ_API_KEY` – Groq LLM API
  * `GOOGLE_API_KEY` – Gemini LLM API
  * `PINECONE_API_KEY` - Pinecone key
  * `PINECONE_ENVIRONMENT`- pinecone environment
* **YAML config:** `src/chatbot_backend/config/agents.yaml` & `tasks.yaml` define agent roles, tools, and expected outputs.


---

## License

This project is **open-source** and available under the MIT License.

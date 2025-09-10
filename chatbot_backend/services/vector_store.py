import json
import os
import glob
import uuid
from typing import List, Dict
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import CohereEmbeddingFunction

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    print("Or set environment variables manually")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

class VectorStoreClient:
    """
    A client to handle all vector database interactions for the chatbot.
    """
    
    def __init__(self, db_path: str = "./data/vector_db"):
        """Initializes the vector database client and embedding model."""
        try:
            if not COHERE_API_KEY:
                print(f"Environment variables available: {list(os.environ.keys())}")
                print(f"COHERE_API_KEY value: {repr(COHERE_API_KEY)}")
                raise ValueError("COHERE_API_KEY environment variable not set.")

            print("Initializing Cohere embedding function...")
            # Use the CohereEmbeddingFunction with explicit model
            self.embedding_function = CohereEmbeddingFunction(
                api_key=COHERE_API_KEY,
                model_name="embed-english-v3.0"  
            )

            print(f"Initializing ChromaDB persistent client at path: {db_path}...")
            self.client = PersistentClient(path=db_path)
            
            try:
                self.collection = self.client.get_collection(
                    name="products_and_faqs",
                    embedding_function=self.embedding_function
                )
                print(f"Using existing vector store collection '{self.collection.name}'.")
            except Exception as e:
                print(f"Collection issue detected: {e}")
                print("Recreating collection with correct embedding dimensions...")
                self._recreate_collection()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VectorStoreClient: {e}")

    def _recreate_collection(self):
        """Delete and recreate the collection to fix dimension mismatches."""
        try:
            # Try to delete the existing collection
            try:
                self.client.delete_collection(name="products_and_faqs")
                print("Deleted existing collection with incorrect dimensions.")
            except Exception as e:
                print(f"No existing collection to delete or deletion failed: {e}")
            
            # Create a new collection
            self.collection = self.client.create_collection(
                name="products_and_faqs",
                embedding_function=self.embedding_function
            )
            print(f"Created new vector store collection '{self.collection.name}' with correct dimensions.")
        except Exception as e:
            raise RuntimeError(f"Failed to recreate collection: {e}")

    def reset_collection(self):
        """Completely reset the collection - useful for debugging."""
        print("Resetting collection...")
        self._recreate_collection()
        print("Collection reset complete.")

    def _process_document(self, file_path: str) -> List[Dict]:
        """
        Processes documents from a given file path, supporting JSON and other formats.
        Returns a list of dictionaries with 'document' and 'metadata'.
        """
        documents_to_ingest = []
        file_extension = os.path.splitext(file_path)[1].lower()
        print(f"Processing file: {file_path}")

        if file_extension == '.json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping '{file_path}'. File is empty or not a valid JSON. Error: {e}")
                return []
            
            for doc in data:
                if not isinstance(doc, dict):
                    print(f"Skipping non-dictionary item in {file_path}: {doc}")
                    continue
                
                doc_type = doc.get("type", "unknown")
                text_content = ""
                metadata = {}
                
                if doc_type == "product":
                    text_content = (
                        f"Product Name: {doc.get('name', '')}. "
                        f"Description: {doc.get('description', '')}. "
                        f"Category: {doc.get('category', 'General')}. "
                        f"Price: {doc.get('price', 'N/A')}. "
                        f"Attributes: {json.dumps(doc.get('attributes', {}))}"
                    )
                    metadata = {
                        "source": "product",
                        "name": doc.get('name', ''),
                        "category": doc.get('category', 'General'),
                        "price": doc.get('price', 'N/A'),
                    }
                elif doc_type == "faq":
                    text_content = f"Question: {doc.get('question', '')}. Answer: {doc.get('answer', '')}"
                    metadata = {
                        "source": "faq",
                        "category": doc.get('category', 'General'),
                    }
                
                if text_content:
                    documents_to_ingest.append({
                        "document": text_content,
                        "metadata": metadata,
                    })
        else:
            print(f"Warning: Unsupported file type '{file_extension}'. Skipping.")
            
        return documents_to_ingest

    def ingest_data(self, knowledge_dir: str):
        """Loads and ingests all supported documents from a directory into the vector store."""
        print(f"Starting data ingestion from directory: {knowledge_dir}")

        print("Clearing existing data from the vector store...")
        try:
            ids = self.collection.get(limit=999999)['ids']
            if ids:
                self.collection.delete(ids=ids)
                print("Successfully cleared all existing data.")
            else:
                print("Collection was already empty. No data to clear.")
        except Exception as e:
            print(f"Failed to clear existing data: {e}. Attempting to proceed with ingestion.")
        
        all_documents = []
        for file_path in glob.glob(os.path.join(knowledge_dir, "**/*.json"), recursive=True):
            if os.path.isfile(file_path):
                all_documents.extend(self._process_document(file_path))

        if not all_documents:
            print("No documents found to ingest.")
            return

        print(f"Processing and preparing {len(all_documents)} documents for ingestion...")
        
        documents = [doc['document'] for doc in all_documents]
        metadatas = [doc['metadata'] for doc in all_documents]
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            print(f"Successfully ingested {len(documents)} documents into the vector store.")
        except Exception as e:
            if "dimension" in str(e).lower():
                print(f"Dimension mismatch detected: {e}")
                print("Attempting to recreate collection with correct dimensions...")
                self._recreate_collection()
                # Try again after recreating
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                )
                print(f"Successfully ingested {len(documents)} documents into the recreated vector store.")
            else:
                raise e

    def query(self, query_text: str, n_results: int = 5, where_filter: dict = None) -> List[Dict]:
        """
        Queries the vector store for similar documents with optional metadata filtering.
        """
        print(f"Querying vector store for: '{query_text}' with filter: {where_filter}")
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter,
                include=['metadatas', 'documents', 'distances']
            )
            
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": results["distances"][0][i]
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"An error occurred during vector store query: {e}")
            return []

if __name__ == "__main__":
    try:
        client = VectorStoreClient()
        knowledge_dir = "./knowledge"
        
        # Uncomment this line if you want to force reset the collection
        # client.reset_collection()
        
        client.ingest_data(knowledge_dir)
        
        print("\nTesting Query...")
        product_results = client.query("velvet armchair dimensions")
        if product_results:
            print("Query Successful. Found relevant documents.")
            for result in product_results[:2]:  # Show first 2 results
                print(f"Score: {result['score']:.3f} - {result['document'][:100]}...")
        else:
            print("Query failed or no documents found.")

    except Exception as e:
        print(f"A fatal error occurred: {e}")
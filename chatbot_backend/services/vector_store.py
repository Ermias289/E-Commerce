import json
import os
import glob
import uuid
import shutil
import tempfile
import time
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
    
    def __init__(self, db_path: str = "./data/vector_db", force_clean: bool = None):
        """
        Initializes the vector database client and embedding model.
        
        Args:
            db_path: Path to the ChromaDB database
            force_clean: If True, deletes the entire DB directory before initialization.
        """
        self.db_path = db_path
        self.max_retries = 3
        
        try:
            if not COHERE_API_KEY:
                print(f"Environment variables available: {list(os.environ.keys())}")
                print(f"COHERE_API_KEY value: {repr(COHERE_API_KEY)}")
                raise ValueError("COHERE_API_KEY environment variable not set.")

            # Auto-detect deployment environment and force clean
            if force_clean is None:
                # Check for deployment environment indicators
                is_deployment = (
                    os.getenv('RAILWAY_ENVIRONMENT') is not None or
                    os.getenv('RENDER') is not None or
                    os.getenv('HEROKU_APP_NAME') is not None or
                    os.getenv('VERCEL') is not None or
                    os.getenv('FORCE_CLEAN_CHROMADB', '').lower() == 'true' or
                    os.getenv('NODE_ENV') == 'production'
                )
                force_clean = is_deployment
            
            if force_clean:
                print("Deployment environment detected - cleaning ChromaDB directory...")
                self._aggressive_cleanup()

            print("Initializing Cohere embedding function...")
            self.embedding_function = CohereEmbeddingFunction(
                api_key=COHERE_API_KEY,
                model_name="embed-english-v3.0"  
            )

            # Try to initialize with retries
            self._initialize_with_retries()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VectorStoreClient after all retries: {e}")

    def _aggressive_cleanup(self):
        """Aggressively clean the database directory with multiple strategies."""
        print(f"Starting aggressive cleanup of {self.db_path}...")
        
        # Strategy 1: Standard directory removal
        if os.path.exists(self.db_path):
            try:
                shutil.rmtree(self.db_path)
                print("✓ Standard directory cleanup successful")
            except Exception as e:
                print(f"✗ Standard cleanup failed: {e}")
        
        # Strategy 2: Force cleanup with file-by-file removal
        if os.path.exists(self.db_path):
            try:
                self._force_remove_directory(self.db_path)
                print("✓ Force file-by-file cleanup successful")
            except Exception as e:
                print(f"✗ Force cleanup failed: {e}")
        
        # Strategy 3: Use temporary directory as new location
        if os.path.exists(self.db_path):
            print("Directory still exists, using temporary location...")
            self.db_path = os.path.join(tempfile.gettempdir(), f"chromadb_{int(time.time())}")
            print(f"New database path: {self.db_path}")
        
        # Strategy 4: Ensure directory is completely clean
        os.makedirs(self.db_path, exist_ok=True)
        
        # Wait a moment for file system to sync
        time.sleep(0.5)

    def _force_remove_directory(self, path):
        """Force remove directory by changing permissions and removing files individually."""
        import stat
        
        def handle_remove_readonly(func, path, exc):
            if os.path.exists(path):
                os.chmod(path, stat.S_IWRITE)
                func(path)
        
        for root, dirs, files in os.walk(path, topdown=False):
            # Remove files
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.chmod(file_path, stat.S_IWRITE)
                    os.remove(file_path)
                except:
                    pass
            
            # Remove directories
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                try:
                    os.chmod(dir_path, stat.S_IWRITE)
                    os.rmdir(dir_path)
                except:
                    pass
        
        # Finally remove the root directory
        try:
            os.chmod(path, stat.S_IWRITE)
            os.rmdir(path)
        except:
            pass

    def _initialize_with_retries(self):
        """Initialize ChromaDB client with multiple retry strategies."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"Initialization attempt {attempt + 1}/{self.max_retries}")
                print(f"Using ChromaDB path: {self.db_path}")
                
                # Create fresh client instance
                self.client = PersistentClient(path=self.db_path)
                
                # Try to create collection directly (skip get_collection check)
                collection_name = f"products_and_faqs_v{int(time.time())}"  # Use unique name
                try:
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function,
                        get_or_create=True  # This will get if exists or create if not
                    )
                    print(f"✓ Successfully created collection: {collection_name}")
                    return  # Success!
                
                except Exception as collection_error:
                    print(f"Collection creation failed: {collection_error}")
                    
                    # Try with get_or_create=False
                    try:
                        # Delete any existing collections first
                        collections = self.client.list_collections()
                        for col in collections:
                            try:
                                self.client.delete_collection(col.name)
                                print(f"Deleted existing collection: {col.name}")
                            except:
                                pass
                        
                        # Create new collection
                        self.collection = self.client.create_collection(
                            name=collection_name,
                            embedding_function=self.embedding_function
                        )
                        print(f"✓ Successfully created collection after cleanup: {collection_name}")
                        return  # Success!
                    
                    except Exception as retry_error:
                        last_error = retry_error
                        print(f"Retry failed: {retry_error}")
                
            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1} failed: {e}")
                
                # If this isn't the last attempt, try more aggressive cleanup
                if attempt < self.max_retries - 1:
                    print(f"Attempting more aggressive cleanup for retry {attempt + 2}...")
                    
                    # Use a completely new path
                    self.db_path = os.path.join(tempfile.gettempdir(), f"chromadb_retry_{attempt}_{int(time.time())}")
                    print(f"Switching to new path: {self.db_path}")
                    os.makedirs(self.db_path, exist_ok=True)
                    
                    # Wait before retry
                    time.sleep(1)
        
        # If we get here, all attempts failed
        raise RuntimeError(f"All {self.max_retries} initialization attempts failed. Last error: {last_error}")

    def _recreate_collection(self):
        """Delete and recreate the collection to fix dimension mismatches."""
        try:
            # Generate unique collection name
            collection_name = f"products_and_faqs_v{int(time.time())}"
            
            # Try to delete any existing collections
            try:
                collections = self.client.list_collections()
                for col in collections:
                    try:
                        self.client.delete_collection(col.name)
                        print(f"Deleted existing collection: {col.name}")
                    except Exception as e:
                        print(f"Failed to delete collection {col.name}: {e}")
            except Exception as e:
                print(f"Failed to list collections: {e}")
            
            # Create a new collection with unique name
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Created new vector store collection '{collection_name}'.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to recreate collection: {e}")

    def reset_collection(self):
        """Completely reset the collection - useful for debugging."""
        print("Resetting collection...")
        self._recreate_collection()
        print("Collection reset complete.")

    def force_clean_and_reinitialize(self):
        """Force clean the entire database and reinitialize from scratch."""
        print("Force cleaning database and reinitializing...")
        try:
            self._aggressive_cleanup()
            self._initialize_with_retries()
            print("Database successfully cleaned and reinitialized.")
        except Exception as e:
            raise RuntimeError(f"Failed to clean and reinitialize database: {e}")

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

    def ingest_data(self, knowledge_dir: str, clean_before_ingest: bool = False):
        """
        Loads and ingests all supported documents from a directory into the vector store.
        
        Args:
            knowledge_dir: Directory containing the knowledge files
            clean_before_ingest: If True, completely clean the database before ingesting
        """
        print(f"Starting data ingestion from directory: {knowledge_dir}")

        if clean_before_ingest:
            print("Cleaning database before ingestion...")
            self.force_clean_and_reinitialize()

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
        # The client will automatically clean in deployment environments
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
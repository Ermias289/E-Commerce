import json
import os
import glob
import uuid
import time
from typing import List, Dict, Optional
import cohere
from pinecone import Pinecone, ServerlessSpec, PodSpec

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    print("Or set environment variables manually")

# Environment variables
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = "luxe"

class VectorStoreClient:
    """
    A client to handle all vector database interactions using Pinecone.
    """
    
    def __init__(self, index_name: str = INDEX_NAME, dimension: int = 1024):
        """
        Initializes the Pinecone vector database client.
        
        Args:
            index_name: Name of the Pinecone index
            dimension: Dimension of the embeddings (Cohere embed-english-v3.0 uses 1024)
        """
        self.index_name = index_name
        self.dimension = dimension
        
        try:
            if not COHERE_API_KEY:
                raise ValueError("COHERE_API_KEY environment variable not set.")
            if not PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY environment variable not set.")

            print("Initializing Cohere client...")
            self.cohere_client = cohere.Client(COHERE_API_KEY)

            print("Initializing Pinecone client...")
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            
            self._initialize_index()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PineconeVectorStoreClient: {e}")

    def _initialize_index(self):
        """Initialize or connect to the Pinecone index."""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=PINECONE_ENVIRONMENT
                    )
                )
                time.sleep(5)
            else:
                index_info = self.pc.describe_index(self.index_name)
                current_dimension = index_info.dimension
                if current_dimension != self.dimension:
                    print(f"Dimension mismatch detected: existing index has dimension {current_dimension}, expected {self.dimension}.")
                    print(f"Deleting and recreating index {self.index_name} to fix dimension.")
                    self.pc.delete_index(self.index_name)
                    
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region=PINECONE_ENVIRONMENT
                        )
                    )
                    time.sleep(5)

                print(f"Using existing Pinecone index: {self.index_name}")
            
            self.index = self.pc.Index(self.index_name)
            
            stats = self.index.describe_index_stats()
            print(f"Index stats: {stats.get('total_vector_count', 0)} vectors")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone index: {e}")

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Cohere."""
        try:
            response = self.cohere_client.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            
            return [list(map(float, emb)) for emb in response.embeddings]
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise

    def _generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for a query."""
        try:
            response = self.cohere_client.embed(
                texts=[query_text],
                model="embed-english-v3.0",
                input_type="search_query"
            )
            
            embedding = response.embeddings[0]
            return list(map(float, embedding))
                
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            raise

    def clear_index(self):
        """Clear all vectors from the index."""
        try:
            print("Clearing all vectors from the index...")
            self.index.delete(delete_all=True)
            print("Successfully cleared all vectors from the index.")
            time.sleep(2)
        except Exception as e:
            print(f"Failed to clear index: {e}")

    def _process_document(self, file_path: str) -> List[Dict]:
        """
        Processes documents from a given file path, supporting JSON formats.
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
                        "price": str(doc.get('price', 'N/A')),
                        "type": "product"
                    }
                elif doc_type == "faq":
                    text_content = f"Question: {doc.get('question', '')}. Answer: {doc.get('answer', '')}"
                    metadata = {
                        "source": "faq",
                        "category": doc.get('category', 'General'),
                        "type": "faq"
                    }
                
                if text_content:
                    documents_to_ingest.append({
                        "document": text_content,
                        "metadata": metadata,
                    })
        else:
            print(f"Warning: Unsupported file type '{file_extension}'. Skipping.")
            
        return documents_to_ingest

    def ingest_data(self, knowledge_dir: str, clear_before_ingest: bool = False, batch_size: int = 100):
        """
        Loads and ingests all supported documents from a directory into Pinecone.
        
        Args:
            knowledge_dir: Directory containing the knowledge files
            clear_before_ingest: If True, clear the index before ingesting
            batch_size: Number of vectors to upsert in each batch
        """
        print(f"Starting data ingestion from directory: {knowledge_dir}")

        if clear_before_ingest:
            self.clear_index()

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
        
        print("Generating embeddings...")
        embeddings = self._generate_embeddings(documents)
        
        vectors = []
        for i, (doc, metadata, embedding) in enumerate(zip(documents, metadatas, embeddings)):
            vector_id = str(uuid.uuid4())
            metadata['text'] = doc
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
        
        print(f"Upserting {len(vectors)} vectors to Pinecone...")
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
                print(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            except Exception as e:
                print(f"Error upserting batch {i//batch_size + 1}: {e}")
                raise

        print("Waiting for Pinecone to index new data...")
        time.sleep(10)

        print(f"Successfully ingested {len(vectors)} documents into Pinecone index '{self.index_name}'.")

    def query(self, query_text: str, n_results: int = 5, filter_dict: Optional[Dict] = None, where_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Queries the Pinecone index for similar documents with optional metadata filtering.
        """
        active_filter = where_filter if where_filter is not None else filter_dict
        print(f"Querying Pinecone for: '{query_text}' with filter: {active_filter}")
        
        try:
            query_embedding = self._generate_query_embedding(query_text)
            
            response = self.index.query(
                vector=query_embedding,
                top_k=n_results,
                include_metadata=True,
                filter=active_filter
            )
            
            formatted_results = []
            for match in response.matches:
                formatted_results.append({
                    "id": match.id,
                    "document": match.metadata.get('text', ''),
                    "metadata": {k: v for k, v in match.metadata.items() if k != 'text'},
                    "score": float(match.score)
                })
            
            print(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            print(f"An error occurred during Pinecone query: {e}")
            return []

    def get_index_stats(self) -> Dict:
        """Get statistics about the index."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "index_fullness": stats.get('index_fullness', 0)
            }
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {}

    def delete_vectors(self, vector_ids: List[str]):
        """Delete specific vectors by ID."""
        try:
            self.index.delete(ids=vector_ids)
            print(f"Deleted {len(vector_ids)} vectors")
        except Exception as e:
            print(f"Error deleting vectors: {e}")

if __name__ == "__main__":
    try:
        client = VectorStoreClient()
        
        stats = client.get_index_stats()
        print(f"Current index stats: {stats}")
        
        knowledge_dir = "./knowledge"
        
        client.ingest_data(knowledge_dir, clear_before_ingest=True)
        
        expected_count = 13
        print("Waiting for vectors to be indexed...")
        max_retries = 5
        retries = 0
        while True:
            updated_stats = client.get_index_stats()
            if updated_stats['total_vectors'] >= expected_count:
                print(f"Indexing complete. Total vectors: {updated_stats['total_vectors']}")
                break
            
            print(f"Current vector count: {updated_stats['total_vectors']}. Retrying in 10 seconds...")
            time.sleep(10)
            retries += 1
            if retries >= max_retries:
                print("Max retries reached. Continuing anyway.")
                break
        
        print(f"Updated index stats: {updated_stats}")
        
        print("\nTesting Query...")
        product_results = client.query("velvet armchair dimensions")
        if product_results:
            print("Query Successful. Found relevant documents.")
            for result in product_results[:2]:
                print(f"Score: {result['score']:.3f} - {result['document'][:100]}...")
        else:
            print("Query failed or no documents found.")
            
        print("\nTesting Filtered Query (products only)...")
        filtered_results = client.query("luxury furniture", filter_dict={"source": "product"})
        if filtered_results:
            print("Filtered Query Successful.")
            for result in filtered_results[:2]:
                print(f"Score: {result['score']:.3f} - {result['document'][:100]}...")

    except Exception as e:
        print(f"A fatal error occurred: {e}")
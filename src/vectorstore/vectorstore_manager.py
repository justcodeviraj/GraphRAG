from pathlib import Path
from typing import List, Optional, Dict
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


class VectorStoreManager:
    
    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        persist_directory: Optional[str] = None
    ):
       
        self.embedding_model = embedding_model
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.persist_directory = persist_directory
        self.vectorstore = None
        
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
    
    def create_vectorstore(
        self,
        documents: List[Document],
        collection_name: str = "knowledge_base"
    ) -> Chroma:
        
        # Add unique IDs to documents if not present
        for i, doc in enumerate(documents):
            if 'id' not in doc.metadata:
                doc.metadata['id'] = f"doc_{i}"
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.persist_directory
        )
        
        return self.vectorstore
    
    def load_vectorstore(
        self,
        collection_name: str = "knowledge_base"
    ) -> Chroma:
    
        if not self.persist_directory:
            raise ValueError("persist_directory must be set to load vector store")
        
        if not Path(self.persist_directory).exists():
            raise ValueError(f"Vector store directory does not exist: {self.persist_directory}")
        
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        return self.vectorstore
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None #  Metadata filter
    ) -> List[Document]:
    
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_vectorstore first.")
        
        if filter_dict:
            results = self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple[Document, float]]:
        
        if not self.vectorstore:
            raise ValueError("Vector store not initialized.")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    # Add new documents to existing vector store.
    def add_documents(self, documents: List[Document]) -> List[str]:
    
        if not self.vectorstore:
            raise ValueError("Vector store not initialized.")
        
        # Add unique IDs if not present
        for i, doc in enumerate(documents):
            if 'id' not in doc.metadata:
                doc.metadata['id'] = f"doc_new_{i}"
        
        ids = self.vectorstore.add_documents(documents)
        return ids
    
    def get_relevant_contexts(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.5
    ) -> List[str]:
        
        results = self.similarity_search_with_score(query, k=k)
        
        contexts = []
        for doc, score in results:
            if score <= (1 - score_threshold):  # Lower scores are better in ChromaDB (distance-based)
                contexts.append(doc.page_content)
        
        return contexts
    
    def delete_collection(self, collection_name: str = "knowledge_base") -> None:
        
        if self.vectorstore:
            self.vectorstore.delete_collection()
            self.vectorstore = None
    
    def get_statistics(self) -> Dict:
       
        if not self.vectorstore:
            return {"status": "not initialized"}
        
        collection = self.vectorstore._collection
        count = collection.count()
        
        return {
            "num_documents": count,
            "embedding_model": self.embedding_model,
            "persist_directory": self.persist_directory,
            "collection_name": collection.name
        }



## Specialized vector store for entity embeddings
class EntityVectorStore(VectorStoreManager):
    
    
    def create_entity_vectorstore(
        self,
        entities: List[Dict[str, str]],
        collection_name: str = "entities"
    ) -> Chroma:
        
        # Convert entities to documents
        documents = []
        for i, entity_dict in enumerate(entities):
            entity_name = entity_dict['entity']
            entity_type = entity_dict.get('type', 'UNKNOWN')
            
            doc = Document(
                page_content=entity_name,
                metadata={
                    'id': f"entity_{i}",
                    'entity': entity_name,
                    'type': entity_type
                }
            )
            documents.append(doc)
        
        return self.create_vectorstore(documents, collection_name)
    
    def find_similar_entities(
        self,
        entity: str,
        k: int = 5
    ) -> List[str]:

        results = self.similarity_search(entity, k=k)
        return [doc.metadata.get('entity', doc.page_content) for doc in results]


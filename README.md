# GraphRAG: Knowledge Graph-Enhanced Retrieval Augmented Generation

I implemented a hybrid retrieval system that combines knowledge graphs with vector search for enhanced contextual retrieval. The approach extracts entities and relationships from documents using local LLMs, builds a structured knowledge graph, and retrieves information through both graph traversal and semantic similarity.

## Dataset

The system works with any collection of documents (TXT and PDF formats supported). I designed it primarily for medical and scientific domains where relationships between concepts matter - diseases, symptoms, treatments, and their causal connections. Documents are automatically loaded, chunked, and processed through the indexing pipeline.

```python
from src.kg.text_loader import load_and_split_documents
chunks = load_and_split_documents("data/docs", chunk_size=1000, chunk_overlap=200)
```

Documents are stored locally in `data/docs/` and all generated artifacts (graph, vector store, metadata) are saved to the specified output directory.

## Implementation

I built the system with three main components working together. First, I use Ollama's Llama 3.1 to extract structured entities (diseases, symptoms, treatments, etc.) and their relationships (causes, treats, symptom of, etc.) from each text chunk. This extraction uses carefully designed prompts.

Then I construct a NetworkX directed graph using the extracted data, where entities become nodes and relationships become weighted edges. The graph tracks entity frequencies, supports neighborhood traversal up to configurable depths, and enables subgraph extraction around query-relevant entities.

I embeded all document chunks using nomic-embed-text model and stored in a Chroma vector database. I use these embeddings to do traditional semantic search that complement the graph-based retrieval.

## Retrieval Strategies

I implemented two retrieval approaches that leverage different aspects of the knowledge graph. The LocalGraphRetriever focuses on immediate neighborhoods, it extracts entities from queries, traverses the graph up to 2 hops around those entities, and returns both the graph context and vector search results. This works well for focused questions about specific concepts.

The second approch, GlobalGraphRetriever, takes a broader approach by first detecting communities in the graph using the Louvain algorithm. When a query arrives, it retrieves relevant documents through vector search, identifies which communities contain entities from those documents, and generates summaries of the top relevant communities.


## Repository Structure

```
GraphRAG/
├── data/
│   ├── docs/
│   │   └── sample_medical.txt
│   └── artifacts/
│       ├── graph.gpickle
│       ├── metadata.json
│       └── chroma/
│
├── src/
│   ├── index/
│   │   └── index_runner.py
│   │
│   ├── kg/
│   │   ├── entity_extractor.py
│   │   ├── graph_builder.py
│   │   └── text_loader.py
│   │
│   ├── retriever/
│   │   └── graph_retriever.py
│   │
│   └── vectorstore/
│       └── vectorstore_manager.py
│
├── eval/
│   └── eval_script.py
│
└── requirements.txt 
```

## Running the Code

It requires Ollama installed locally with llama3.1 and nomic-embed-text models. The indexing script handles all processing automatically.

```bash
# To indexe all datafiles
python pipelines/index_runner.py --input data/docs --out artifacts

# Takes 10-15 minutes for typical medical corpus
```

## Usage Example

```python
from src.kg.graph_builder import KnowledgeGraphBuilder
from src.vectorstore.vectorstore_manager import VectorStoreManager
from src.retriever.graph_retriever import LocalGraphRetriever

graph_builder = KnowledgeGraphBuilder.load('artifacts/graph.gpickle')
vectorstore_manager = VectorStoreManager(
    embedding_model='nomic-embed-text',
    persist_directory='artifacts/chroma'
)
vectorstore = vectorstore_manager.load_vectorstore()

retriever = LocalGraphRetriever(
    graph=graph_builder.graph,
    vectorstore=vectorstore,
    max_hops=2
)

results = retriever.retrieve("What causes diabetic neuropathy?", k_vector=5)
print("Graph Context:", results['graph_context'])
print("Vector Context:", results['vector_context'])
```

## Requirements

```
langchain>=0.1.0
langchain-community>=0.0.10
chromadb>=0.4.0
networkx>=3.0
python-louvain>=0.16
ollama>=0.1.0
pypdf>=3.17.0
```

Install with: `pip install -r requirements.txt`

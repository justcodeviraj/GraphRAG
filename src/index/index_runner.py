import argparse
from pathlib import Path
import sys

# Add src to path for importing functions from other folders
sys.path.append(str(Path(__file__).parent.parent.parent))

from kg.text_loader import load_and_split_documents
from kg.entity_extractor import EntityRelationExtractor
from kg.graph_builder import KnowledgeGraphBuilder
from vectorstore.vectorstore_manager import VectorStoreManager


def run_indexing_pipeline(
    input_dir: str,
    output_dir: str,
    llm_model: str = "llama3.1",
    embedding_model: str = "nomic-embed-text",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
   
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    chunks = load_and_split_documents(
        input_dir=input_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    extractor = EntityRelationExtractor(llm_model=llm_model)
    all_entities = []
    all_relations = []
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]        
        entities, relations = extractor.extract_from_documents(batch)
        all_entities.extend(entities)
        all_relations.extend(relations)

    graph_builder = KnowledgeGraphBuilder()
    graph_builder.build_from_extractions(all_entities, all_relations)
    
    stats = graph_builder.get_statistics()
    # print(f"Graph has: {stats['num_nodes']} nodes, and {stats['num_edges']} edges")
    
    graph_path = output_path / "graph.gpickle"
    graph_builder.save(str(graph_path))
    
    vectorstore_manager = VectorStoreManager(
        embedding_model=embedding_model,
        persist_directory=str(output_path / "chroma")
    )
    vectorstore_manager.create_vectorstore(chunks)
    vs_stats = vectorstore_manager.get_statistics()

    metadata = {
        'llm_model': llm_model,
        'embedding_model': embedding_model,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'num_documents': len(chunks),
        'num_entities': len(all_entities),
        'num_relations': len(all_relations),
        'graph_stats': stats
    }
    
    import json # lazy import
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # good to know where everyone is 
    print(f"Artifacts: {output_dir}")
    print(f"\nGraph: {graph_path}")
    print(f"\nVector Store: {output_path / 'chroma'}")
    print(f"\nMetadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Run GraphRAG indexing pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing documents")
    parser.add_argument("--out", type=str, required=True, help="Output directory for artifacts")
    parser.add_argument("--llm", type=str, default="llama3.1", help="Ollama LLM model (default: llama3.1)")
    parser.add_argument("--emb", type=str, default="nomic-embed-text", help="Ollama embedding model (default: nomic-embed-text)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Text chunk size (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap (default: 200)")
    
    args = parser.parse_args()
    
    run_indexing_pipeline(
        input_dir=args.input,
        output_dir=args.out,
        llm_model=args.llm,
        embedding_model=args.emb,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

# test
if __name__ == "__main__":
    main()

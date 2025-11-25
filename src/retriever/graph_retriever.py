from typing import List, Dict, Set, Tuple, Optional
import networkx as nx
from langchain.schema import Document
from langchain_community.llms import Ollama

# For immediate neighborhood of query entities.
class LocalGraphRetriever:
       
    def __init__(
        self,
        graph: nx.DiGraph,
        vectorstore,
        llm_model: str = "llama3.1",
        max_hops: int = 2
    ):
        self.graph = graph
        self.vectorstore = vectorstore
        self.llm = Ollama(model=llm_model, temperature=0.1)
        self.max_hops = max_hops
    
    def extract_query_entities(self, query: str) -> List[str]:
        prompt = f"""Extract the main medical entities (diseases, symptoms, treatments, etc.) from this question.
                    Return ONLY a comma-separated list of entities, nothing else.
                    Question: {query}
                    Entities:"""
                            
        response = self.llm.invoke(prompt)
        entities = [e.strip() for e in response.split(',') if e.strip()]
        return entities[:5]  # Limit to top 5 entities 
    


    def get_graph_context(self, entities: List[str]) -> str:
        context_parts = []
        all_nodes = set()
        
        for entity in entities:
            entity_normalized = entity.lower().strip()
            if not self.graph.has_node(entity_normalized):
                continue
            all_nodes.add(entity_normalized)
            
            # Get neighbors
            current_level = {entity_normalized}
            for hop in range(self.max_hops):
                next_level = set()
                for node in current_level:
                    successors = set(self.graph.successors(node))
                    predecessors = set(self.graph.predecessors(node))
                    next_level.update(successors | predecessors)
                
                all_nodes.update(next_level)
                current_level = next_level
        
        subgraph = self.graph.subgraph(all_nodes)
        
        for node in all_nodes:
            node_data = self.graph.nodes[node]
            entity_name = node_data.get('original_name', node)
            entity_type = node_data.get('type', 'UNKNOWN')
    
            context_parts.append(f"Entity: {entity_name} (Type: {entity_type})")
            for target in self.graph.successors(node):
                if target in all_nodes:
                    edge_data = self.graph.edges[node, target]
                    relation = edge_data.get('relation', 'RELATED_TO')
                    target_name = self.graph.nodes[target].get('original_name', target)
                    context_parts.append(f" {relation}: {target_name}")
        
        return "\n".join(context_parts) if context_parts else "No graph context found."
    
    def retrieve(
        self,
        query: str,
        k_vector: int = 5,
        k_graph: int = 3
    ) -> Dict[str, any]:
        
        # Vector-based retrieval
        vector_docs = self.vectorstore.similarity_search(query, k=k_vector)
        vector_context = "\n\n".join([doc.page_content for doc in vector_docs])
        
        # Graph-based retrieval
        query_entities = self.extract_query_entities(query)
        graph_context = self.get_graph_context(query_entities)
        
        return {
            'query': query,
            'query_entities': query_entities,
            'vector_context': vector_context,
            'graph_context': graph_context,
            'vector_docs': vector_docs
        }



# To analyze overall graph structure
class GlobalGraphRetriever:
    
    def __init__(
        self,
        graph: nx.DiGraph,
        vectorstore,
        llm_model: str = "llama3.1"
    ):
        
        self.graph = graph
        self.vectorstore = vectorstore
        self.llm = Ollama(model=llm_model, temperature=0.1)
        self.communities = None
        self.community_summaries = {}
    
    def detect_communities(self, resolution: float = 1.0) -> Dict[str, int]:
        
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        import community.community_louvain as community_louvain
        self.communities = community_louvain.best_partition(undirected, resolution=resolution)

        # self.communities = {}
        # for i, component in enumerate(nx.weakly_connected_components(self.graph)):
        #     for node in component:
        #         self.communities[node] = i
        
        return self.communities
    
    def summarize_community(self, community_id: int) -> str:
        
        if community_id in self.community_summaries:
            return self.community_summaries[community_id]
        
        # Get all nodes in community
        community_nodes = [node for node, cid in self.communities.items() if cid == community_id]
        
        if not community_nodes:
            return "Empty community"
        
        # Get node types and relations
        entity_types = {}
        relation_counts = {}
        
        for node in community_nodes:
            node_type = self.graph.nodes[node].get('type', 'UNKNOWN')
            entity_types[node_type] = entity_types.get(node_type, 0) + 1
            
            for _, target, data in self.graph.edges(node, data=True):
                if target in community_nodes:
                    rel = data.get('relation', 'RELATED_TO')
                    relation_counts[rel] = relation_counts.get(rel, 0) + 1
        
        # Create summary
        summary = f"Community {community_id} ({len(community_nodes)} entities):\n"
        summary += f"Entity types: {', '.join([f'{k}({v})' for k, v in entity_types.items()])}\n"
        summary += f"Relations: {', '.join([f'{k}({v})' for k, v in relation_counts.items()])}\n"
        
        # Sample key entities (by frequency/degree)
        top_nodes = sorted(
            community_nodes,
            key=lambda n: self.graph.degree(n) * self.graph.nodes[n].get('frequency', 1),
            reverse=True
        )[:5]
        
        summary += f"Key entities: {', '.join([self.graph.nodes[n].get('original_name', n) for n in top_nodes])}"
        
        self.community_summaries[community_id] = summary
        return summary
    
    def retrieve(
        self,
        query: str,
        k_vector: int = 5,
        k_communities: int = 3
    ) -> Dict[str, any]:
    
        # Ensure communities are detected
        if self.communities is None:
            self.detect_communities()
        
        # Vector-based retrieval
        vector_docs = self.vectorstore.similarity_search(query, k=k_vector)
        vector_context = "\n\n".join([doc.page_content for doc in vector_docs])
        
        # Get entities from top vector results
        relevant_entities = set()
        for doc in vector_docs[:3]:
            # Extract entities mentioned in doc (simple keyword matching)
            content_lower = doc.page_content.lower()
            for node in self.graph.nodes():
                if node in content_lower:
                    relevant_entities.add(node)
        
        # Find relevant communities
        relevant_community_ids = set()
        for entity in relevant_entities:
            if entity in self.communities:
                relevant_community_ids.add(self.communities[entity])
        
        # Get top k communities by size
        community_sizes = {}
        for cid in relevant_community_ids:
            community_sizes[cid] = sum(1 for _, c in self.communities.items() if c == cid)
        
        top_communities = sorted(
            relevant_community_ids,
            key=lambda c: community_sizes.get(c, 0),
            reverse=True
        )[:k_communities]
        
        # Generate community summaries
        community_contexts = []
        for cid in top_communities:
            summary = self.summarize_community(cid)
            community_contexts.append(summary)
        
        graph_context = "\n\n".join(community_contexts) if community_contexts else "No relevant communities found."
        
        return {
            'query': query,
            'vector_context': vector_context,
            'graph_context': graph_context,
            'communities_used': top_communities,
            'vector_docs': vector_docs
        }

import pickle
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import networkx as nx
from collections import defaultdict


class KnowledgeGraphBuilder:
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_frequencies = defaultdict(int)
        self.relation_types = set()
    
    ############ Nodes
    def add_entities(self, entities: List[Dict[str, str]]) -> None:
        for entity_dict in entities:
            entity = entity_dict['entity']
            entity_type = entity_dict.get('type', 'UNKNOWN')
            entity_normalized = entity.lower().strip()
            self.entity_frequencies[entity_normalized] += 1  # for graph stats
            
            if self.graph.has_node(entity_normalized):
                self.graph.nodes[entity_normalized]['frequency'] += 1
                if entity_type != 'UNKNOWN':
                    self.graph.nodes[entity_normalized]['type'] = entity_type
            else:
                self.graph.add_node(
                    entity_normalized,
                    type=entity_type,
                    frequency=1,
                    original_name=entity
                )
    
    ############# Edges
    def add_relations(self, relations: List[Dict[str, str]]) -> None:
        for rel_dict in relations:
            source = rel_dict['source'].lower().strip()
            target = rel_dict['target'].lower().strip()
            relation = rel_dict['relation'].upper().strip()
            self.relation_types.add(relation)
            
            # nodes (entities) should exist in graph first
            if not self.graph.has_node(source):
                self.graph.add_node(source, type='UNKNOWN', frequency=1, original_name=rel_dict['source'])
            if not self.graph.has_node(target):
                self.graph.add_node(target, type='UNKNOWN', frequency=1, original_name=rel_dict['target'])
            
            # now we can add an edge/relation (if any)
            if self.graph.has_edge(source, target):
                self.graph.edges[source, target]['weight'] += 1
            else:
                self.graph.add_edge(
                    source,
                    target,
                    relation=relation,
                    weight=1
                )
    

    ########### graph
    def build_graph(self, entities: List[Dict[str, str]], relations: List[Dict[str, str]]) -> None:
        self.add_entities(entities)
        self.add_relations(relations)
    
    
    ########### DFS in graph
    def get_neighbors(self, entity: str, depth: int = 1) -> Set[str]:
        entity_normalized = entity.lower().strip()
        
        if not self.graph.has_node(entity_normalized): # just to be safe
            return set()

        neighbors = set()
        current_level = {entity_normalized}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            
            neighbors.update(next_level)
            current_level = next_level
        
        return neighbors - {entity_normalized} # only want neighbors
    
    
    ############# A subgraph having current node and neighbor nodes
    def get_subgraph(self, entities: List[str], depth: int = 1) -> nx.DiGraph:
        all_nodes = set()
        for entity in entities:
            entity_normalized = entity.lower().strip()
            if self.graph.has_node(entity_normalized):
                all_nodes.add(entity_normalized)
                all_nodes.update(self.get_neighbors(entity_normalized, depth))

        return self.graph.subgraph(all_nodes).copy()
    

    ################ Helper function to understand if DFS results make any sense 
    def get_entity_context(self, entity: str, max_hops: int = 2) -> str:
    
        entity_normalized = entity.lower().strip()
        
        if not self.graph.has_node(entity_normalized): # why do i keep checking this idk
            return f"Entity '{entity}' not found."
        
        context_parts = []
        node_data = self.graph.nodes[entity_normalized]
        
        context_parts.append(
            f"{node_data.get('original_name', entity)} (Type: {node_data.get('type', 'UNKNOWN')}, "
            f"Frequency: {node_data.get('frequency', 1)})"
        )
        
        for target in self.graph.successors(entity_normalized):
            edge_data = self.graph.edges[entity_normalized, target]
            relation = edge_data.get('relation', 'RELATED_TO')
            target_name = self.graph.nodes[target].get('original_name', target)
            context_parts.append(f"{relation} is related to {target_name}")
        
        for source in self.graph.predecessors(entity_normalized):
            edge_data = self.graph.edges[source, entity_normalized]
            relation = edge_data.get('relation', 'RELATED_TO')
            source_name = self.graph.nodes[source].get('original_name', source)
            context_parts.append(f"{source_name} is related to {relation}")
        
        return "\n".join(context_parts)
    

    ########### overall structure of graph
    def get_statistics(self) -> Dict:
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_relation_types': len(self.relation_types),
            'relation_types': list(self.relation_types),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
    
    def save(self, output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)
        
        print(f"Graph was saved to {output_path}") # very underrated line. 
    
    @classmethod
    def load(cls, input_path: str) -> 'KnowledgeGraphBuilder':
        with open(input_path, 'rb') as f:
            graph = pickle.load(f)
        
        builder = cls()
        builder.graph = graph
        
        # rebuild 
        for node in graph.nodes():
            builder.entity_frequencies[node] = graph.nodes[node].get('frequency', 1)
        
        for _, _, data in graph.edges(data=True):
            if 'relation' in data:
                builder.relation_types.add(data['relation'])
        
        print(f"Graph loaded from {input_path}")
        return builder

# test
if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    
    entities = [
        {"entity": "Diabetes", "type": "DISEASE"},
        {"entity": "Neuropathy", "type": "DISEASE"},
        {"entity": "High blood sugar", "type": "CAUSE"},
        {"entity": "Insulin", "type": "TREATMENT"}
    ]
    
    relations = [
        {"source": "Diabetes", "relation": "CAUSES", "target": "Neuropathy"},
        {"source": "High blood sugar", "relation": "CAUSES", "target": "Neuropathy"},
        {"source": "Insulin", "relation": "TREATS", "target": "Diabetes"}
    ]
    
    builder.build_from_extractions(entities, relations)
    
    #Graph stats
    for key, value in builder.get_statistics().items():
        print(f"  {key}: {value}")
    
    #Context 
    print(builder.get_entity_context("Diabetes"))

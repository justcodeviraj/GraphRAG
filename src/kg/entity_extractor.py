import json
import re
from typing import List, Dict, Tuple
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import Document


class EntityRelationExtractor:
    
    def __init__(self, llm_model: str = "llama3.1", temperature: float = 0.1):
        
        self.llm = Ollama(model=llm_model, temperature=temperature)
        self.entity_prompt = self._entity_prompt() # prompted gpt to get prompts :)
        self.relation_prompt = self._relation_prompt() # prompted gpt to get prompts :)
    
    def _entity_prompt(self) -> PromptTemplate:
        
        template = """You are an expert at extracting entities from medical and scientific text.

                    Extract ALL important entities from the following text. Focus on:
                    - Medical conditions and diseases
                    - Symptoms
                    - Treatments and medications
                    - Body parts and organs
                    - Medical procedures
                    - Risk factors
                    - Causes and pathogens

                    Text: {text}

                    Return your answer as a JSON list of entities with their types. Format:
                    [
                    {{"entity": "entity name", "type": "DISEASE|SYMPTOM|TREATMENT|ANATOMY|PROCEDURE|RISK_FACTOR|CAUSE"}},
                    ...
                    ]

                    Entities:"""
        return PromptTemplate(template=template, input_variables=["text"])
    
    def _relation_prompt(self) -> PromptTemplate:

        template = """You are an expert at extracting relationships between medical entities.

                    Given the following text and entities, extract relationships between them.

                    Text: {text}

                    Entities: {entities}

                    Extract relationships like:
                    - CAUSES (e.g., "diabetes CAUSES neuropathy")
                    - TREATS (e.g., "insulin TREATS diabetes")
                    - SYMPTOM_OF (e.g., "numbness SYMPTOM_OF neuropathy")
                    - LOCATED_IN (e.g., "tumor LOCATED_IN brain")
                    - INCREASES_RISK (e.g., "smoking INCREASES_RISK cancer")

                    Return your answer as a JSON list. Format:
                    [
                    {{"source": "entity1", "relation": "CAUSES", "target": "entity2"}},
                    ...
                    ]

                    Relationships:"""
        return PromptTemplate(
            template=template,
            input_variables=["text", "entities"]
        )
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
    
        prompt = self.entity_prompt.format(text=text)
        response = self.llm.invoke(prompt)
        entities = self._parse_json_response(response) # since llm outputs are messy
        
        # Validate and clean entities
        valid_entities = []
        for ent in entities:
            if isinstance(ent, dict) and 'entity' in ent:
                valid_entities.append({
                    'entity': ent['entity'].strip(),
                    'type': ent.get('type', 'UNKNOWN').strip()
                })
        return valid_entities
        
    
    def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        
        if not entities: # if llm betrayed in finding any
            return []
        
        entity_str = json.dumps(entities, indent=2)
        prompt = self.relation_prompt.format(text=text, entities=entity_str)
        response = self.llm.invoke(prompt)
        relations = self._parse_json_response(response)
        
        # Validate and clean relations
        valid_relations = []
        entity_names = {e['entity'].lower() for e in entities}
        for rel in relations:
            if isinstance(rel, dict) and all(k in rel for k in ['source', 'relation', 'target']):
                
                # Check if relations are based only on given entities and not some other 
                source = rel['source'].strip()
                target = rel['target'].strip()
                if source.lower() in entity_names and target.lower() in entity_names:
                    valid_relations.append({
                        'source': source,
                        'relation': rel['relation'].strip(),
                        'target': target
                    })
        
        return valid_relations
    
    def _parse_json_response(self, response: str) -> List[Dict]:
    
        # Try to find JSON array in response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try parsing entire response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        return []
    
    def extract_from_document(
        self,
        document: Document
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    
        text = document.page_content
        entities = self.extract_entities(text)
        relations = self.extract_relations(text, entities)
        
        return entities, relations
    
    # for more than one document
    def extract_from_documents(
        self,
        documents: List[Document]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        
        all_entities = []
        all_relations = []
        
        for i, doc in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}...")
            entities, relations = self.extract_from_document(doc)
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        print(f"\nExtracted {len(all_entities)} entities and {len(all_relations)} relations")
        return all_entities, all_relations

# test
if __name__ == "__main__":
   
    extractor = EntityRelationExtractor(llm_model="llama3.1")
    
    sample_text = """
    Diabetic neuropathy is nerve damage caused by diabetes. High blood sugar 
    levels can injure nerves throughout the body. Common symptoms include 
    numbness and tingling in the feet. Treatment involves managing blood sugar 
    levels with insulin and medications.
    """
    
    entities = extractor.extract_entities(sample_text)
    print("Entities:", json.dumps(entities, indent=2))
    
    relations = extractor.extract_relations(sample_text, entities)
    print("\nRelations:", json.dumps(relations, indent=2))
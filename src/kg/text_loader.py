import os
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentLoader:
    
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        
    def load_documents(self) -> List[Document]:
        documents = []
        
        # text files
        txt_loader = DirectoryLoader(
            str(self.input_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents.extend(txt_loader.load())
        
        # pdf files
        for pdf_file in self.input_dir.rglob("*.pdf"):
            pdf_loader = PyPDFLoader(str(pdf_file))
            documents.extend(pdf_loader.load())
        
        return documents


class TextSplitter:
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None # user can choose based on data they upload
    ):
    
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""] # should be enough for clean data
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:

        return self.splitter.split_documents(documents) # not a recursion lol. its langchain's custom split_documents


def load_and_split_documents(
    input_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:

    loader = DocumentLoader(input_dir)
    documents = loader.load_documents()
    
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    
    return chunks

# test
if __name__ == "__main__":
    chunks = load_and_split_documents("data/docs")
    print("chunk preview:")
    print(chunks[0].page_content[:200] if chunks else "No chunks found")
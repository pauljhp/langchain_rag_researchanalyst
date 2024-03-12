# document retrival agent

from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
import chromadb
from typing import Literal, List, Dict
import numpy as np
import os


embedding_model = AzureOpenAIEmbeddings(model=os.environ.get("DEFAULT_EMBEDDING_MODEL"))

chroma_client = chromadb.HttpClient(
    host=os.environ.get("CHROMADB_ENDPOINT"), 
    port=os.environ.get("CHROMADB_PORT"))
VectorStore = Literal["chroma", "neo4j"]


def chroma_retrieve_documents(
        db_name: str,
        query: str,
        n_results: int,
        filter: Dict,
        embedding_model=embedding_model
        ) -> List[Dict]:
    collection = chroma_client.get_collection(db_name)
    query_embedding = embedding_model.embed_query(query)
    res = collection.query(
        query_embedding,
        where=filter,
        n_results=n_results
    )
    return res
    
class Retriever:
    # FIXME - change to fetch from environment vars
    llm_16k = AzureChatOpenAI(
        deployment_name="gpt-35-16k", 
        model_name="gpt-35-turbo-16k", 
        api_version="2023-07-01-preview"
        )
    
    def __init__(
            self,
            database_name: str, 
            vector_store: VectorStore="chroma",
            filter: Dict={},
            top_k: int=30
            ):
        match vector_store:
            case "chroma":
                db = Chroma(
                    client=chroma_client, 
                    collection_name=database_name, 
                    embedding_function=embedding_model)
            case _:
                raise NotImplementedError


        self.qa_chain_16k = RetrievalQA.from_chain_type(
            self.llm_16k, 
            retriever=db.as_retriever(
                search_kwargs={
                    'filter': filter,
                    'k': top_k,
                }), 
            verbose=True, 
            return_source_documents=True, 
            input_key="query"
            )

    def get_answer(
            self,
            question: str):
        res = self.qa_chain_16k(
        {"query": 
        question})
        answer = {
            "answer": res.get("result"),
            "sources": res.get("source_documents")
        }
        return answer
    
# def 
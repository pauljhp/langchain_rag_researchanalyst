# document retrival agent

from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
import chromadb
from typing import Literal, List, Dict
import numpy as np

embedding_model = AzureOpenAIEmbeddings(model="text-embedding")

chroma_client = chromadb.HttpClient(
    host="https://impax-chromadb-test.azurewebsites.net/", 
    port="8000")
VectorStore = Literal["chroma", "neo4j"]

class Retriever:
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
            "sources": np.unique(res.get("source_documents")
            )
        }
        return answer
    
# def 
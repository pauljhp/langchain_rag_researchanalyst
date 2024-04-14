# document retrival agent

from langchain_community.vectorstores import Chroma, Qdrant
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import (
    RetrievalQA,
    StuffDocumentsChain,
    LLMChain,
    ReduceDocumentsChain,
    MapReduceDocumentsChain,
)
from langchain_community.document_transformers import LongContextReorder
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ChatMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.memory import BaseMemory
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate, 
    MessagesPlaceholder
)
from typing import Literal, List, Dict, Optional, Tuple
import drivers
import utils
from qdrant_client.http.models import (
    Filter, 
    # FieldCondition, MatchValue, Range, DatetimeRange, ValuesCount
    )


def chroma_retrieve_documents(
        db_name: str,
        query: str,
        n_results: int,
        filter: Dict,
        embedding_model=drivers.EmbeddingModel.default_embedding_model
        ) -> List[Dict]:
    collection = drivers.VectorDBClients.chroma_client.get_collection(db_name)
    query_embedding = embedding_model.embed_query(query)
    res = collection.query(
        query_embedding,
        where=filter,
        n_results=n_results
    )
    return res

def qdrant_retrieve_documents(
        db_name: str,
        query: str,
        n_results: int,
        filter: Optional[Filter]=None,
        embedding_model=drivers.EmbeddingModel.default_embedding_model
    ):
    query_embedding = embedding_model.embed_query(query)
    client = drivers.VectorDBClients.qdrant_client
    search_res = client.search(
        collection_name=db_name,
        query_vector=query_embedding,
        query_filter=filter,
        limit=n_results
    )
    return search_res
    
class Retriever:
    """Retrieve relevant documents from the vector database and reason over 
    the context"""
    # FIXME - change to fetch from environment vars
    llm_16k = AzureChatOpenAI(
        deployment_name="gpt-35-16k", 
        model_name="gpt-35-turbo-16k", 
        api_version="2023-07-01-preview"
        )
    
    def _get_memory(self, **kwargs) -> BaseMemory:
        memory = ConversationBufferMemory(**kwargs)
        return memory

    def _get_reduce_documents_chain(
        self,
        question: str,
        memory):
        """question-aware map-reduce document chain"""
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
            )
        document_variable_name = "context"
        # TODO - pass in question to prompt
        summarizer_prompt = PromptTemplate.from_template(
            "Summarize this context: {context} "
            "This is for answering question {question}",
            partial_variables={"question": question}
            )
        reducer_prompt = PromptTemplate.from_template(
            "combine these summaries: {context}"
            "This is for answering question {question}",
            partial_variables={"question": question}
            )
        document_chain = LLMChain(
            llm=self.llm_16k,
            prompt=summarizer_prompt,
            memory=memory
            )
        reduce_llm_chain = LLMChain(
            llm=self.llm_16k, 
            prompt=reducer_prompt,
            memory=memory
            )
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name
            )
        reduce_documents_chain = ReduceDocumentsChain(
            token_max=12000,
            combine_documents_chain=combine_documents_chain,
            )
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=document_chain,
            reduce_documents_chain=reduce_documents_chain,
            )
        return map_reduce_chain

    def __init__(
            self,
            vector_store: drivers.VectorDBTypes="qdrant",
            database_name: Optional[str]=None, 
            filter: Dict={},
            top_k: int=30
            ):
        match vector_store:
            case "chroma":
                self.db = Chroma(
                    client=drivers.VectorDBClients.chroma_client, 
                    collection_name=database_name, 
                    embeddings=drivers.EmbeddingModel.default_embedding_model)
            case "qdrant":
                self.db = Qdrant(
                    client=drivers.VectorDBClients.qdrant_client,
                    collection_name=database_name,
                    embeddings=drivers.EmbeddingModel.default_embedding_model,
                    )
            case "azuresearch":
                self.db = drivers.VectorDBClients.azure_search_client_rh # Azure search takes an endpoint instead of a client object
            case _:
                raise NotImplementedError
        self.memory = self._get_memory()
        self.filter = filter
        self.top_k = top_k
        self.qa_chain_16k = RetrievalQA.from_chain_type(
            self.llm_16k, 
            retriever=self.db.as_retriever(
                search_kwargs={
                    'filter': filter,
                    'k': top_k,
                }), 
            verbose=True, 
            return_source_documents=True, 
            input_key="query"
            )

    def _get_summarized_context(
            self,
            question: str
        ):
        reordering = LongContextReorder() # Important! 
        docs = self.db.search(question, "similarity", filter=self.filter, k=self.top_k)
        reordered_docs = reordering.transform_documents(docs)
        self.map_reduce_chain_ = self._get_reduce_documents_chain(
            question=question, 
            memory=self.memory)
        condensed_docs = self.map_reduce_chain_.combine_docs(
            reordered_docs, token_max=12000
        )
        page_conent, metadata = condensed_docs
        metadata.update(utils.combine_metadata([doc.metadata for doc in docs]))
        condensed_docs = utils.create_document(
            page_content=page_conent,
            metadata=metadata
        )
        condensed_docs.metadata = utils.combine_metadata([doc.metadata for doc in docs])
        return condensed_docs
    
    def get_recursive_retrieval_agent(
            self, question: str,
        ) -> Tuple[Runnable, ChatMessageHistory]:
        """Use this for when the context window is expected to be long"""
        # TODO - change to conversation style
        context = ChatMessageHistory()
        summary = self._get_summarized_context(question)
        context.add_user_message("Base your answer on the following information: "
                                 f"<info> {summary.page_content} </info>")
        context.add_user_message("If asked, your information source is:"
                            f"<sources>: {summary.metadata} </sources>")
        prompt = ChatPromptTemplate.from_messages(
            [("system", 
              "You are an assistant good at synthecizing information. "
              "Use and only use the information provided to you to answer questions. "
              ),
              MessagesPlaceholder(variable_name="context"),
              ("human", "{input}")]
        )
        runnable = prompt | self.llm_16k
        return runnable, context

    def get_combined_answer(self, 
            question: str,
            # return_source: bool=False,
            chat_history: Optional[ChatMessageHistory]=None,):
        runnable, context = self.get_recursive_retrieval_agent(question)
        if chat_history is not None:
            context.add_messages(chat_history.messages)
        answer = runnable.invoke(
            {"context": context.messages, "input": question}
        )
        return answer
    
    def get_answer(
            self,
            question: str):
        """Plain vanilla QA chain. Use it when your context window fits into
        a sngle model"""
        res = self.qa_chain_16k(
        {"query": 
        question})
        answer = {
            "answer": res.get("result"),
            "sources": res.get("source_documents")
        }
        return answer
    
# def 
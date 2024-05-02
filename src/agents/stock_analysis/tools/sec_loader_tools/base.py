import concurrent.futures
import json
import os
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional

from llama_index.core.readers.base import BaseReader
from .sec_filings import SECExtractor
from llama_index.core.node_parser import (
    SentenceSplitter, TokenTextSplitter,
)
from llama_index.core import (
    VectorStoreIndex, ServiceContext, PromptHelper, StorageContext
)
from llama_index.core.schema import (
    TextNode, NodeRelationship, RelatedNodeInfo, Document
) 
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
import itertools
import utils


embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name=os.environ.get("DEFAULT_EMBEDDING_MODEL"),
        # tokenizer="cl100k_base"
        )
llm = AzureOpenAI(
    deployment_name="gpt-35-16k", 
    model_name="gpt-35-turbo-16k", 
    api_version="2023-07-01-preview",
    # tokenizer="cl100k_base"
    )

transformations = [
    TokenTextSplitter(chunk_size=512,
                      chunk_overlap=512//10,
                    #   tokenizer="cl100k_base"
                      )
]

prompt_helper = PromptHelper(
    context_window=16832, 
    num_output=1500, 
    chunk_size_limit=512,
    # tokenizer="cl100k_base"
    )

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    prompt_helper=prompt_helper, 
    )


class SECFilingsLoader(BaseReader):
    """
    SEC Filings loader
    Get the SEC filings of multiple tickers.
    This will load the relevant data into a dedicated on-disk vector store
    """
    accepted_filing_types = ["10-K", "10-Q"]

    def extractor_fn(self, inputs: Dict[str, str]):
        return SECExtractor(
                    [self.ticker], 
                    self.amount, 
                    filing_type=inputs["filing_type"], 
                    include_amends=self.include_amends
                ).get_text_from_url(inputs["url"])

    def __init__(
        self,
        ticker: str,
        collection_name: Optional[str]=None,
        amount: int=3,
        # filing_type: str = "10-K",
        num_workers: int = 4,
        include_amends: bool = False,
        vector_store_type: utils.driver.VectorDBTypes="chromadb",
    ):
        if collection_name is None:
            collection_name = ticker
        # assert filing_type in [
        #     "10-K",
        #     "10-Q",
        # ], "The supported document types are 10-K and 10-Q"

        self.ticker = ticker
        self.amount = amount
        self.num_workers = num_workers
        self.include_amends = include_amends
        self.extractors = [SECExtractor(
                    [ticker], 
                    amount, 
                    filing_type=filing_type, 
                    include_amends=include_amends
            ) for filing_type in self.accepted_filing_types] 
        self.vector_store = utils.CustomQueryEngine()\
            .get_vector_store(
                collection_name=collection_name,
                vector_store_type=vector_store_type
                )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
            )
        # os.makedirs("data", exist_ok=True)

    def multiprocess_run(self, tic) -> List[Document]:
        # print(f"Started for {tic}")
        tic_dict = {self.ticker: []}
        for se in self.extractors:
            d = se.get_accession_numbers(tic) # tic_dict schema: {"ticker": [{"year": xx, "accession_number": xx}]}
            new_ls = d.get(self.ticker)
            if isinstance(new_ls, list): tic_dict[self.ticker] += new_ls
        # text_dict = defaultdict(list)
        for tic, fields in tic_dict.items():
            # os.makedirs(f"data/{tic}", exist_ok=True)
            print(f"Started for {tic}")
            filing_type_mapping = {4: "10-K", 6: "10-Q"}
            field_filingtype_urls = [
                (filing_type_mapping.get(len(field["year"])), field["url"]) 
                for field in fields]
            years = [field.get("year") for field in fields]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                results = executor.map(
                    self.extractor_fn, 
                    [dict(filing_type=ft, url=url) 
                     for ft, url in 
                     field_filingtype_urls
                     ]
                     )
            for idx, res in enumerate(results):
                print(idx)
                all_text, filing_type = res
                documents = [
                    Document(
                        metadata={
                            "title": title,
                            "filing_type": filing_type,
                            "year": years[idx],
                            "url": field_filingtype_urls[idx][1],
                            "ticker": tic
                        },
                        text=f"{title}\n====\n{text}",
                    )
                     for title, text in all_text.items()
                ]
                
        return documents

    def load_data(self) -> VectorStoreIndex:
        start = time.time()
        documents = self.multiprocess_run(self.ticker)
        # with concurrent.futures.ThreadPoolExecutor(
        #     max_workers=thread_workers
        # ) as executor:
        #     results = executor.map(self.multiprocess_run, self.ticker)
        # documents = list(itertools.chain(*results))
        # print(len(documents), type(documents[0]), documents[0])
        index = VectorStoreIndex.from_documents(
                    embed_model=embed_model,
                    documents=documents,
                    service_context=service_context,
                    transformations=transformations,
                    storage_context=self.storage_context
                )
        return index
    

        for res in results:
            curr_tic = next(iter(res.keys()))
            for data in res[curr_tic]:
                curr_year = data["year"]
                curr_filing_type = data["filing_type"]
                if curr_filing_type in ["10-K/A", "10-Q/A"]:
                    curr_filing_type = curr_filing_type.replace("/", "")
                if curr_filing_type in ["10-K", "10-KA"]:
                    os.makedirs(f"data/{curr_tic}/{curr_year}", exist_ok=True)
                    with open(
                        f"data/{curr_tic}/{curr_year}/{curr_filing_type}.json", "w"
                    ) as f:
                        json.dump(data, f, indent=4)
                elif curr_filing_type in ["10-Q", "10-QA"]:
                    os.makedirs(f"data/{curr_tic}/{curr_year[:-2]}", exist_ok=True)
                    with open(
                        f"data/{curr_tic}/{curr_year[:-2]}/{curr_filing_type}_{curr_year[-2:]}.json",
                        "w",
                    ) as f:
                        json.dump(data, f, indent=4)
                print(
                    f"Done for {curr_tic} for document {curr_filing_type} and year"
                    f" {curr_year}"
                )

        print(f"It took {round(time.time()-start,2)} seconds")

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import BaseTool

from src.constants import RETRIEVER_TOOL

from loguru import logger
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Type

import sklearn

class Retrieve(BaseModel):
    """Represents the context based on the user's uploaded file and Elasticsearch configuration."""
    query: str = Field(description="The query string derived from the user's conversation")
    is_macos: bool = Field(description="Indicates if the system is macOS.")
    es_url: str = Field(description="The URL of the Elasticsearch instance to query")
    es_index: str = Field(description="The name of the Elasticsearch index to query")
    content_field: str = Field(description="The field name in the Elasticsearch index that contains the content")
    knn_field: str = Field(description="The field name in the Elasticsearch index that contains the KNN data")
    k: int = Field(description="The number of nearest neighbors to retrieve for KNN")

class RetrieverTool(BaseTool):
    """
    Represents a tool for retrieving information from the RAG model.

    Attributes:
        name (str): The name of the tool, set to "RAG retriever".
        description (str): A description of the tool's purpose.
        args_schema (Type[BaseModel]): The schema for the arguments expected by the tool.
        return_direct (bool): Indicates whether the tool returns its result directly.
    """
    name: str = RETRIEVER_TOOL
    description: str = "This tool is used to retrieve information from the RAG model."
    args_schema: Type[BaseModel] = Retrieve
    return_direct: bool = True
    
    def _set_encoder(self, is_macos: bool) -> HuggingFaceEmbeddings:
        """
        Sets the encoder for the RetrieverTool based on the macOS flag.

        This method returns a HuggingFaceEmbeddings instance configured with a specific model name and model kwargs. 
        If the system is macOS, it sets the working memory to 2048 before creating the encoder instance.

        Args:
            is_macos (bool): Indicates if the system is macOS.

        Returns:
            HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings with the specified configuration.
        """
        model_kwargs = {
            "model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "model_kwargs": {
                "tokenizer_kwargs": {
                    "clean_up_tokenization_spaces": False
                },
                "device": "cpu"
            }
        }
        
        if is_macos:
            with sklearn.config_context(working_memory=2048):
                return HuggingFaceEmbeddings(**model_kwargs)
        else:
            return HuggingFaceEmbeddings(**model_kwargs)
        
    
    def _hybrid_query(self, 
                      is_macos: bool,
                      k: int, 
                      knn_field: str, 
                      query: str) -> Dict[str, Any]:
        """
        Constructs a hybrid query for Elasticsearch that combines a text-based search with a KNN search.

        This method sets up an encoder, embeds the query string, 
        and then constructs a hybrid query that includes both a text-based search and a KNN search. 
        
        The text-based search is filtered by the provided title IDs, and 
        the KNN search is performed on the specified field with the embedded query vector. 
        The size of the result set is set to the specified value of k.

        Args:
            k (int): The number of results to return.
            knn_field (str): The field name in the Elasticsearch index for the KNN search.
            query (str): The query string to embed and use for the search.
            title_ids (List[str]): A list of title IDs to filter the search results by.

        Returns:
            Dict[str, Any]: A dictionary representing the hybrid query to be executed against Elasticsearch.
        """
        logger.info(f"Loading encoder..")
        encoder = self._set_encoder(is_macos)
        logger.info(f"Loading encoder.. done")
        
        logger.info(f"Encoding the query as vector..")
        query_vector = encoder.embed_query(query)
        logger.info(f"Encoding the query as vector.. done")
        
        return {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["document^3"],
                            "type": "best_fields",
                            "boost": 0.5
                        }
                    }
                }
            },
            "knn": {
                "field": knn_field,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": 10000,
                "boost": 0.5
            },
            "size": k
        }
    
    def _es_hybrid_search(self, query: str,
                          is_macos: bool,
                          es_url: str,
                          es_index: str,
                          content_field: str,
                          knn_field: str,
                          k: int) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search using Elasticsearch, combining text-based and KNN search methods.

        This method constructs a hybrid query that combines a text-based search with a KNN search. 
        It then executes this query against an Elasticsearch index, filtering the results by a set of title IDs. 
        The method iteratively fetches results until it reaches the desired number of unique results, 
        specified by the parameter `k`.

        Args:
            query (str): The query string to use for the search.
            es_url (str): The URL of the Elasticsearch instance.
            es_index (str): The name of the Elasticsearch index to search.
            content_field (str): The field name in the Elasticsearch index for the text-based search.
            knn_field (str): The field name in the Elasticsearch index for the KNN search.
            k (int): The number of unique results to return.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a search result. 
                                  Each dictionary contains fields such as menu_name, access_menu, 
                                  ai_description, description, is_dashboard, is_hidden, 
                                  long_description, keywords, and url.
        """
        logger.info("Setting up hybrid query..")
        query_body = self._hybrid_query(is_macos, k, knn_field, query)
        logger.info("Setting up hybrid query.. done")
        
        search_results = []
            
        try:
            logger.info("Retriving the answer from Elasticsearch..")
            hybrid_retriever = ElasticsearchRetriever.from_es_params(
                index_name=es_index,
                body_func=lambda query: query_body,
                content_field=content_field,
                url=es_url,
            )
            
            hybrid_results = hybrid_retriever.invoke(query)
        except Exception as e:
            raise Exception(f"An error occurred during the hybrid search: {e}")
        finally:
            logger.info("Retriving the answer from Elasticsearch.. done")
        
        for result in hybrid_results:
            search_results.append({
                "document": result.metadata['_source']['document']
            })
        
        logger.info(f"{len(search_results)} found.")
            
        return search_results
    
    def _run(self,
             query: str,
             domain_url: str,
             title_ids: List[str],
             is_macos: bool,
             es_url: str,
             es_index: str,
             content_field: str,
             knn_field: str,
             k: int,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Dict[str, Any]]:
        """
        Executes the hybrid search process using Elasticsearch.

        This method orchestrates the hybrid search process, which combines traditional search with KNN search.
        It utilizes the `_es_hybrid_search` method to perform the actual search and returns the results.

        Args:
            query (str): The search query.
            domain_url (str): The base URL for constructing URLs in the search results.
            title_ids (List[str]): A list of title IDs to filter the search results.
            is_macos (bool): A flag indicating if the search is for macOS.
            es_url (str): The URL of the Elasticsearch instance.
            es_index (str): The name of the Elasticsearch index to search.
            content_field (str): The field name in the Elasticsearch index for the text-based search.
            knn_field (str): The field name in the Elasticsearch index for the KNN search.
            k (int): The number of unique results to return.
            run_manager (Optional[CallbackManagerForToolRun], optional): An optional callback manager for tool run. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a search result.
        """
        return self._es_hybrid_search(query, domain_url, title_ids, 
                                      is_macos, es_url, es_index, 
                                      content_field, knn_field, k)

    async def _arun(self,
                    query: str,
                    domain_url: str,
                    title_ids: List[str],
                    is_macos: bool,
                    es_url: str,
                    es_index: str,
                    content_field: str,
                    knn_field: str,
                    k: int,
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> List[Dict[str, Any]]:
        """
        Executes the hybrid search process asynchronously using Elasticsearch.

        This method orchestrates the hybrid search process asynchronously, combining traditional search with KNN search.
        It utilizes the `_run` method to perform the actual search and returns the results.

        Args:
            query (str): The search query.
            domain_url (str): The base URL for constructing URLs in the search results.
            title_ids (List[str]): A list of title IDs to filter the search results.
            is_macos (bool): A flag indicating if the search is for macOS.
            es_url (str): The URL of the Elasticsearch instance.
            es_index (str): The name of the Elasticsearch index to search.
            content_field (str): The field name in the Elasticsearch index for the text-based search.
            knn_field (str): The field name in the Elasticsearch index for the KNN search.
            k (int): The number of unique results to return.
            run_manager (Optional[AsyncCallbackManagerForToolRun], optional): An optional callback manager for tool run. 
                                                                              Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a search result.
        """
        return self._run(query, domain_url, 
                         title_ids, is_macos, 
                         es_url, es_index, 
                         content_field, knn_field, 
                         k, run_manager=run_manager.get_sync())
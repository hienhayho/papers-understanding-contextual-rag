from llama_index.core.bridge.pydantic import Field, BaseModel


class Settings(BaseModel):
    """
    Settings for the contextual RAG.

    Attributes:
        chunk_size (int): Default chunk size
        service (str): The LLM service, e.g., "openai"
        model (str): The LLM model name, e.g., "gpt-4o-mini"
        original_rag_collection_name (str): The original RAG collection name
        contextual_rag_collection_name (str): The contextual RAG collection name
        qdrant_host (str): The Qdrant host
        qdrant_port (int): The Qdrant port
        elastic_search_url (str): The ElasticSearch URL
        elastic_search_index_name (str): The ElasticSearch index name
        num_chunks_to_recall (int): The number of chunks to recall
        semantic_weight (float): The semantic weight
        bm25_weight (float): The BM25 weight
        top_n (int): Top n documents after reranking
    """

    chunk_size: int = Field(description="The chunk size", default=1024)

    service: str = Field(description="The LLM service", default="openai")

    model: str = Field(description="The LLM model", default="gpt-4o-mini")

    original_rag_collection_name: str = Field(
        description="The original RAG collection name", default="original_rag"
    )

    contextual_rag_collection_name: str = Field(
        description="The contextual RAG collection name", default="contextual_rag"
    )

    qdrant_host: str = Field(description="The Qdrant host", default="localhost")
    qdrant_port: int = Field(description="The Qdrant port", default=6333)

    elastic_search_url: str = Field(
        description="The Elastic URL", default="http://localhost:9200"
    )
    elastic_search_index_name: str = Field(
        description="The Elastic index name", default="contextual_rag"
    )
    num_chunks_to_recall: int = Field(
        description="The number of chunks to recall", default=150
    )

    # Reference: https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb
    semantic_weight: float = Field(description="The semantic weight", default=0.8)
    bm25_weight: float = Field(description="The BM25 weight", default=0.2)

    top_n: int = Field(description="Top n documents after reranking", default=3)


setting = Settings()

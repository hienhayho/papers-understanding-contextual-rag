from llama_index.core.bridge.pydantic import Field
from llama_index.core.bridge.pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service: str = Field(description="The LLM service", default="openai")

    model: str = Field(description="The LLM model", default="gpt-4o-mini")

    original_rag_collection_name: str = Field(
        description="The original RAG collection name", default="original_rag"
    )

    contextual_rag_collection_name: str = Field(
        description="The contextual RAG collection name", default="papers"
    )

    qdrant_host: str = Field(description="The Qdrant host", default="localhost")
    qdrant_port: int = Field(description="The Qdrant port", default=6333)

    elastic_search_url: str = Field(
        description="The Elastic URL", default="http://localhost:9200"
    )


setting = Settings()

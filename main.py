import argparse
import threading
import qdrant_client
from typing import Literal
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core import VectorStoreIndex, StorageContext, Settings

from src.settings import setting

GREEN = "\033[92m"
RESET = "\033[0m"

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--q", type=str, help="Query to search for")
parser.add_argument(
    "--type",
    choices=["origin", "contextual", "both"],
    required=True,
)
args = parser.parse_args()

llm = OpenAI(model="gpt-4o-mini")
Settings.llm = llm


def get_qdrant_vector_store_index(
    client: qdrant_client.QdrantClient, collection_name: str
) -> BaseQueryEngine:
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context
    ).as_query_engine()


def get_index(type: Literal["origin", "contextual", "both"]):
    client = qdrant_client.QdrantClient(
        host=setting.qdrant_host, port=setting.qdrant_port
    )
    if type == "origin":
        return get_qdrant_vector_store_index(
            client=client, collection_name=setting.original_rag_collection_name
        )
    elif type == "contextual":
        return get_qdrant_vector_store_index(
            client=client, collection_name=setting.contextual_rag_collection_name
        )
    elif type == "both":
        return {
            "origin": get_qdrant_vector_store_index(
                client=client, collection_name=setting.original_rag_collection_name
            ),
            "contextual": get_qdrant_vector_store_index(
                client=client, collection_name=setting.contextual_rag_collection_name
            ),
        }


if __name__ == "__main__":
    index = get_index(args.type)

    if args.type == "origin":
        result = index.query(args.q)
        print(f"{GREEN}Origin RAG: {RESET}{result}")

    elif args.type == "contextual":
        result = index.query(args.q)
        print(f"{GREEN}Contextual RAG: {RESET}{result}")

    elif args.type == "both":
        thread = [
            threading.Thread(
                target=lambda: print(
                    f"\n\n{GREEN}Origin RAG: {RESET}{index['origin'].query(args.q)}"
                )
            ),
            threading.Thread(
                target=lambda: print(
                    f"\n\n{GREEN}Contextual RAG: {RESET}{index['contextual'].query(args.q)}"
                )
            ),
        ]

        for t in thread:
            t.start()

        for t in thread:
            t.join()

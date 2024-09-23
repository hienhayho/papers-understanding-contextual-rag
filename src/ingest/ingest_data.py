import os
import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent.parent))

import qdrant_client
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings

from src.settings import setting
from src.ingest.prompt import CONTEXTUAL_PROMPT
from src.readers.paper_reader import llama_parse_read_paper

load_dotenv()

embed_model = OpenAIEmbedding()
Settings.embed_model = embed_model

llm = OpenAI(model=setting.model)
Settings.llm = llm

splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)


def load_parser():
    parser = argparse.ArgumentParser(description="Ingest data into the index")
    parser.add_argument(
        "--folder_dir",
        type=str,
        help="Path to the folder containing the documents",
    )
    parser.add_argument(
        "--type",
        choices=["origin", "contextual"],
        required=True,
    )
    return parser.parse_args()


def split_document(
    document: Document | list[Document],
    show_progress: bool = True,
) -> list[Document]:
    if isinstance(document, Document):
        document = [document]

    nodes = splitter.get_nodes_from_documents(document, show_progress=show_progress)

    return [Document(text=node.get_content()) for node in nodes]


def add_contextual_content(
    single_document: Document,
) -> list[Document]:
    whole_document = single_document.text
    splited_documents = split_document(single_document)
    documents: list[Document] = []

    for chunk in splited_documents:
        messages = [
            ChatMessage(
                role="system",
                content="You are a helpful assistant.",
            ),
            ChatMessage(
                role="user",
                content=CONTEXTUAL_PROMPT.format(
                    WHOLE_DOCUMENT=whole_document, CHUNK_CONTENT=chunk.text
                ),
            ),
        ]
        response = llm.chat(messages)
        new_chunk = response.message.content + "\n\n" + chunk.text
        documents.append(Document(text=new_chunk))

    return documents


def ingest_data(
    documents: list[Document],
    show_progress: bool = True,
    type: Literal["origin", "contextual"] = "contextual",
):
    client = qdrant_client.QdrantClient(
        host=setting.qdrant_host, port=setting.qdrant_port
    )

    if type == "origin":
        collection_name = setting.original_rag_collection_name
    else:
        collection_name = setting.contextual_rag_collection_name

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=show_progress
    )

    return index


def get_contextual_documents(folder_dir: str | Path) -> list[Document]:
    raw_documents = llama_parse_read_paper(folder_dir)

    documents = []

    for raw_document in tqdm(raw_documents):
        documents.extend(add_contextual_content(raw_document))

    return documents


def get_origin_documents(folder_dir: str | Path) -> list[Document]:
    raw_documents = llama_parse_read_paper(folder_dir)

    documents = split_document(raw_documents)

    return documents


def get_documents(folder_dir: str | Path, type: str) -> list[Document]:
    if type == "contextual":
        return get_contextual_documents(folder_dir)
    else:
        return get_origin_documents(folder_dir)


if __name__ == "__main__":
    args = load_parser()
    folder_dir = args.folder_dir

    documents = get_documents(folder_dir, args.type)

    ingest_data(documents, type=args.type)

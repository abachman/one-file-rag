from datetime import datetime
from functools import cache
from typing import List, Dict, TypedDict
import asyncio
import hashlib
import os
import sys

# libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import semantic_search
import anthropic
import numpy as np
import ollama

DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
)
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(DATA_DIR, ".cache"))


# custom type used in semantic search and reranking
class Score(TypedDict):
    corpus_id: int
    score: float
    cross_score: float


@cache
def load_sentence_transformer():
    print(f"> loading SentenceTransformer model")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1",
        trust_remote_code=True,
        cache_folder=CACHE_DIR,
        device="mps",
    )
    print(f"> model.device: {model.device}")
    return model


@cache
def load_cross_encoder():
    print(f"> loading CrossEncoder model")
    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_dir=CACHE_DIR,
        device="mps",
    )
    return model


def generate_documents() -> List[Document]:
    dir = DATA_DIR
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40,
    )

    contents = []
    metas = []
    fcount = 0
    for f in (f for f in os.listdir(dir) if f.endswith(".md")):
        metadata = {
            "filename": os.path.join(dir, f),
        }
        with open(metadata["filename"]) as f:
            content = f.read()
            metadata["md5"] = hashlib.md5(content.encode("utf-8")).hexdigest()
            metas.append(metadata)
            contents.append(content)
            fcount += 1

    documents = splitter.create_documents(contents, metas)
    print(f"> chunked {fcount} files into {len(documents)} documents")

    return documents


def generate_embeddings(
    documents: List[Document],
) -> np.ndarray:
    model = None

    # group documents by .metadata["filename"]
    grouped_documents: Dict[str, List[Document]] = {}
    for doc in documents:
        filename = doc.metadata["filename"]
        if filename not in grouped_documents:
            grouped_documents[filename] = []
        grouped_documents[filename].append(doc)

    start = datetime.now()
    embeddings = np.ndarray((0, 768), dtype=np.float32)
    for filename, docs in grouped_documents.items():
        cache_path = os.path.join(
            CACHE_DIR, f"{filename}.{docs[0].metadata['md5']}.pkl"
        )

        if os.path.exists(cache_path) and not os.environ.get("FORCE_REGENERATE", False):
            print(f"> loading cached embeddings for {docs[0].metadata['filename']}")
            with open(cache_path, "rb") as f:
                doc_embeddings = np.load(f, allow_pickle=True)
                embeddings = np.concatenate((embeddings, doc_embeddings))
        else:
            if model is None:
                model = load_sentence_transformer()
            print(f"> generating embeddings for {docs[0].metadata['filename']}")
            sentences = [f"search_document: {doc.page_content}" for doc in docs]
            doc_embeddings = model.encode(
                sentences, convert_to_numpy=True, show_progress_bar=True
            )
            with open(cache_path, "wb") as f:
                doc_embeddings.dump(f)
                embeddings = np.concatenate((embeddings, doc_embeddings))
    print(
        f"> returning corpus_embeddings {embeddings.shape} in {datetime.now() - start}"
    )
    return embeddings


def refine_query(query: str) -> str:
    print(f'> refining original query: "{query}"')
    prompt = f"Without adding explanation, summarize the following query for use as a search string which retains the original meaning:\n\n  {query}"
    response = ollama.generate(
        model="llama3.1:latest",
        prompt=prompt,
        stream=False,
    )
    print(f"> refined search query for embedding: {response["response"]}")
    return response["response"]


def rerank_candidates(
    original_query: str,
    documents: List[Document],
    cosine_scores: List[Score],
) -> List[Score]:
    """
    Reranking generates a normalized "score" value for each (query, document)
    pair in the initial semantic search results. When we add that score to the
    original cosine similarity score record, we can reorder the semantic search
    results based on CrossEncoder's score.
    """
    cross_encoder = load_cross_encoder()
    cross_input = [
        [original_query, documents[hit["corpus_id"]].page_content]
        for hit in cosine_scores
    ]

    cross_scores = cross_encoder.predict(cross_input, show_progress_bar=True)

    # sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        cosine_scores[idx]["cross_score"] = cross_scores[idx]

    # pick the best 5
    return sorted(cosine_scores, key=lambda x: x["cross_score"], reverse=True)[:5]


def rag_prompt(original_query: str, documents: List[Document]) -> str:
    return "\n".join(
        line.strip()
        for line in f"""
            You are an assistant for question-answering tasks.
            Use the following documents to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise:
            Question: {original_query}
            Documents: {"\n\n".join(doc.page_content for doc in documents)}
            Answer:
        """.splitlines()
    )


def display_ollama_response(prompt: str):
    for part in ollama.generate(
        model="llama3.1:latest",
        prompt=prompt,
        system="You are a helpful assistant who wants to answer questions to the best of your abilities based on the information provided in this conversation.",
        stream=True,
    ):
        print(part["response"], end="", flush=True)
    print()


async def display_anthropic_response(prompt: str):
    api_key = os.environ.get("ANTHROPIC_API_KEY", None)
    if not api_key:
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)
    async with client.messages.stream(
        max_tokens=768,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="claude-3-5-sonnet-20240620",
    ) as stream:
        async for message in stream.text_stream:
            print(message, end="", flush=True)
        print()


def rag_query(original_query: str, documents: List[Document]):
    prompt = rag_prompt(original_query, documents_for_context)

    print()
    print()
    print("---- ollama ------------------------------------")
    print()
    display_ollama_response(prompt)
    print()
    print("---- anthropic ---------------------------------")
    print()
    asyncio.run(display_anthropic_response(prompt))
    print()
    print("----------------------------------------")
    print()

    unique_documents = set(doc.metadata["filename"] for doc in documents)
    print()
    print("documents referenced:")
    for filename in unique_documents:
        print(f"- {filename}")
    print()
    print("----------------------------------------")
    print()


if __name__ == "__main__":
    original_query = (
        "What is the most important thing to understand about the subject described?"
    )
    if len(sys.argv) > 1:
        original_query = sys.argv[1]

    # turn a folder of .md files into text chunks
    documents = generate_documents()

    # generate a semantic search index from the given document collection
    corpus_embeddings = generate_embeddings(documents)

    # refine the original query to summarize and clarify for document search
    refined_query = refine_query(original_query)

    model = load_sentence_transformer()
    query_embedding = model.encode(
        [f"search_query: {refined_query}"], convert_to_numpy=True
    )

    print(f"> search with query embedding {query_embedding.shape}")
    top_k_embedded = 30
    top_n_ranked = 5
    cosine_scores = semantic_search(
        query_embedding, corpus_embeddings, top_k=top_k_embedded
    )
    cosine_scores: List[Score] = cosine_scores[0]
    print(f"> best score {cosine_scores[0]}")
    print(f"> top {top_n_ranked} documents of {top_k_embedded}")
    for score in cosine_scores[:top_n_ranked]:
        print(f">   - {score['corpus_id']}")

    # rerank candidates by reranking model
    # => https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb
    print(f"> reranking candidate documents...")
    reranked_scores = rerank_candidates(original_query, documents, cosine_scores)
    print(f"> new top {top_n_ranked} documents after reranking:")
    for score in reranked_scores[:top_n_ranked]:
        print(f">   - {score['corpus_id']}")

    # perform final RAG query
    print("> generate final RAG response...")
    documents_for_context = [documents[hit["corpus_id"]] for hit in reranked_scores]
    rag_query(original_query, documents_for_context)

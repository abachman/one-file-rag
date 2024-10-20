# RAG search in one file

Dependencies:

- `sentence_transformers`
- `langchain_text_splitters`
- `ollama`
- `anthropic` (optional, requires API key)

References consulted:

- nomic-ai/nomic-embed-text-v1.5: model used for generating initial document embeddings
  - https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- cross-encoder/ms-marco-MiniLM-L-6-v2
  - https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
- sentence_transformers: Used for semantic search
  - https://sbert.net/docs/sentence_transformer/usage/usage.html
  - https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb
- langchain_text_splitters: Used for splitting text into sentences
  - https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
- numpy: Used for array operations. It's the native format that sentence-transformers speaks.
  - https://numpy.org/doc/stable/reference/index.html

## Usage

Before running the code below, put one or more plaintext `*.md` files in `data/`. When in doubt, you could always start with [Frankenstein](https://www.gutenberg.org/ebooks/84).

Next, install dependencies:

```console
# https://github.com/python-poetry/poetry
$ poetry install
```

Now you can ask questions about it/them.

```console
# run a search query on your dataset
$ python -m one_file_rag.engine "Did doctor frankentein perform experiments that some may consider strange?"

> chunked 4 files into 2447 documents
> ...
> generating embeddings for /app/one-file-rag/data/frankenstein.md
> Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:20<00:00, 2.39it/s]
> ...
> returning corpus_embeddings (2447, 768) in 0:00:23.538580
> refining original query: "Did doctor frankentein perform experiments that some may consider strange?"
> refined search query for embedding: "Doctor Frankenstein's unusual medical experiments"
> search with query embedding (1, 768)
> best score {'corpus_id': 1038, 'score': 0.5091668367385864}
> top 5 documents
>
> - 1038
> - 1128
> - 1891
> - 2406
> - 99
>   reranking candidate documents...
>   loading CrossEncoder model
>   Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2.76it/s]
>   new top 5 documents after reranking:
> - 1650
> - 1173
> - 1875
> - 1128
> - 1538
>   generate final RAG response...

---- ollama ------------------------------------

Based on the documents provided, it appears that Victor Frankenstein performed experiments that some may consider strange and even horrific. The narrator seems to be shocked by the details of his tale, and one document mentions "the horror of my proceedings" and "anguish" in relation to Frankenstein's work. However, I don't know the specifics of what exactly he did.

---- anthropic ---------------------------------

Yes, Doctor Frankenstein performed experiments that could be considered strange. The documents describe his work as involving "horror" and causing his heart to sicken at times. The experiments seem to have involved creating a living creature, which others found terrifying and surprising.

---

documents referenced:

- /app/one-file-rag/data/frankenstein.md

---

```

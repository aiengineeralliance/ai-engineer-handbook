# Retrival-Augmented Generation (RAG) in 5 minutes

RAG, aka ChatGPT over private data, is the predominant way applications such as ChatPDF.com and the OpenAI Assistants API works. 

By the end of this tutorial, you will have built the simplest possible RAG application in 5 minutes from scratch without any frameworks.

## What is RAG?

Retrieval-Augmented Generation (RAG) combines information retrieval with large language models (LLMs) to 
improve the factual accuracy and relevance of machine-generated text by accessing external databases. 
This approach is particularly useful for generating responses to queries that require current knowledge or 
specific details not contained within the model's pre-existing training data.

At the core of RAG is the idea that while powerful, large language models (like GPT-3) are limited by the knowledge they
were trained on, this limitation can be overcome by integrating a retrieval component, fetching relevant context at query-time 
and populating the LLM prompt with said context. This retrieval step allows the model to access up-to-date and detailed information beyond what
it has learned during its initial training.


## Why RAG?

- **Up-to-Date Information:** Static models may not always have access to the latest information. RAG ensures that the
  generated content can be augmented with the most recent data.
- **Private data:** Sometimes, the context required to answer a question or complete a prompt is not found
  within
  the model's pre-existing knowledge. RAG can pull in the necessary context from external sources.
- **Hallucination mitigation** The generated answer is grounded in data from the retrieval phase, reducing the likelihood of hallucination.
- **Resource Efficiency:** Fine-tuning large language models with new information is computationally expensive and
  time-consuming. RAG offers a more efficient alternative by retrieving information on-the-fly.


## The application
We want to build a Python application that lets us ask ChatGPT questions over our knowledge base (containing 2 documents) and get back answers that are grounded in these 2 documents (and not from ChatGPT's entire knowledge base).

The knowledge base is: 

1. Horses are domesticated mammals known for their strength, speed, and versatility. They have been crucial to human civilization for transportation, agriculture, and recreational activities. Horses belong to the Equidae family and are herbivores with a digestive system adapted to grazing. Common breeds include the Arabian, Thoroughbred, and Clydesdale.

2. Zebras are African equids known for their distinctive black and white striped coat patterns. They belong to the genus Equus, which also includes horses and donkeys. Zebras are herbivores and primarily graze on grasses. 


!!! note 
    In reality, if the knowledge base only comprised these 2 documents, we wouldn't bother to RAG as these are small enough to fit into the context. However, we're using this simple case to demonstrate how RAG works in general. 


## Install dependencies
```bash
pip install openai chromadb
```

[Sign-up for an OpenAI API key](https://platform.openai.com/docs/quickstart?context=python) if you haven't already.

[ChromaDB](https://www.trychroma.com/) is the vector database we're using. 

## Initialization

```python
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from openai import OpenAI

OPENAI_API_KEY = "sk-xxx" # CHANGEME!
chroma_collection = "animals"
embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path="vector_store")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
```

## Vector store indexing

Given a question, we need to retrieve relevant documents that are then populated into the LLM prompt. 

We'll be using a local vector database (Chroma) and OpenAI vector embeddings to perform this retrieval phase, but the general method applies irrespective of the choice of vector database or embeddings. 

!!! note 
    It is important to use the exact same vector embeddings when indexing and retrieving documents from the vector database.

Before retrieving the documents, we first have to add them to Chroma.

```python
def add_docs_to_vector_store(documents: list[str],
                             metadatas: list[dict] = None, ids: list[str] = None,
                             ):
    collection: Collection = chroma_client.get_or_create_collection(name=chroma_collection,
                                                                    embedding_function=embedding_function)
    if ids is None:
        ids = [str(counter) for counter, doc in enumerate(documents, 1)]
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
```

## The RAG step proper

The RAG step involves retrieving matching documents from Chroma, then populating the prompt with the retrieved documents.

```python
def rag(query: str):
    collection: Collection = chroma_client.get_or_create_collection(name=chroma_collection,
                                                                    embedding_function=embedding_function)
    docs = collection.query(query_texts=query, n_results=5)
    prompt = retrieval_prompt(docs["documents"][0], query)
    return llm_call(prompt, "You are a helpful AI assistant.", model="gpt-3.5-turbo", temperature=0)

def retrieval_prompt(docs, query) -> str:
    return f"""Answer the QUESTION using only the CONTEXT given, nothing else. 
Do not make up an answer, say 'I don't know' if you are not sure. Be succinct.
QUESTION: {query}
CONTEXT: {[doc for doc in docs]}
ANSWER:
"""
```

The retrieval prompt instructs ChatGPT to answer based on the provided context.

The `llm_call()` function is a utility method that calls the completion OpenAI endpoint.

```python
def llm_call(prompt, system_prompt="", **kwargs) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    kwargs["messages"] = messages
    response = openai_client.chat.completions.create(**kwargs)
    generated_texts = [
        choice.message.content.strip() for choice in response.choices
    ]
    return " ".join(generated_texts)
```


## The main() function

```python
if __name__ == "__main__":
    add_docs_to_vector_store([
        "Horses are domesticated mammals known for their strength, speed, and versatility. They have been crucial to human civilization for transportation, agriculture, and recreational activities. Horses belong to the Equidae family and are herbivores with a digestive system adapted to grazing. Common breeds include the Arabian, Thoroughbred, and Clydesdale.",
        "Zebras are African equids known for their distinctive black and white striped coat patterns. They belong to the genus Equus, which also includes horses and donkeys. Zebras are herbivores and primarily graze on grasses. "])
    print(rag("What genus do horses belong to?"))
    print(rag("What genus do zebras belong to?"))
    print(rag("Are zebras and horses of the same genus?"))
    print(rag("What genus do snakes belong to?"))

# Horses belong to the genus Equus.
# Zebras belong to the genus Equus.
# Yes.
# I don't know.    
```

## The entire file

```python
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.api.models import Collection
from openai import OpenAI

OPENAI_API_KEY = "sk-xxx" # CHANGEME
chroma_collection = "animals"
embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path="vector_store")
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def add_docs_to_vector_store(documents: list[str],
                             metadatas: list[dict] = None, ids: list[str] = None,
                             ):
    collection: Collection = chroma_client.get_or_create_collection(name=chroma_collection,
                                                                    embedding_function=embedding_function)
    if ids is None:
        ids = [str(counter) for counter, doc in enumerate(documents, 1)]
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )


def retrieval_prompt(docs, query) -> str:
    return f"""Answer the QUESTION using only the CONTEXT given, nothing else. 
Do not make up an answer, say 'I don't know' if you are not sure. Be succinct.
QUESTION: {query}
CONTEXT: {[doc for doc in docs]}
ANSWER:
"""


def rag(query: str):
    collection: Collection = chroma_client.get_or_create_collection(name=chroma_collection,
                                                                    embedding_function=embedding_function)
    docs = collection.query(query_texts=query, n_results=2)
    prompt = retrieval_prompt(docs["documents"][0], query)
    return llm_call(prompt, "You are a helpful AI assistant.", model="gpt-3.5-turbo", temperature=0.0)


def llm_call(prompt, system_prompt="", **kwargs) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    kwargs["messages"] = messages
    response = openai_client.chat.completions.create(**kwargs)
    generated_texts = [
        choice.message.content.strip() for choice in response.choices
    ]
    return " ".join(generated_texts)


if __name__ == "__main__":
    add_docs_to_vector_store([
        "Horses are domesticated mammals known for their strength, speed, and versatility. They have been crucial to human civilization for transportation, agriculture, and recreational activities. Horses belong to the Equidae family and are herbivores with a digestive system adapted to grazing. Common breeds include the Arabian, Thoroughbred, and Clydesdale.",
        "Zebras are African equids known for their distinctive black and white striped coat patterns. They belong to the genus Equus, which also includes horses and donkeys. Zebras are herbivores and primarily graze on grasses. "])
    print(rag("What genus do horses belong to?"))
    print(rag("What genus do zebras belong to?"))
    print(rag("Are zebras and horses of the same genus?"))
    print(rag("What genus do snakes belong to?"))
```

## Where to from here?

Congratulations! You've built your first RAG application. 

For the sake of brevity, conciseness and ease of comprehension, we've omitted all kinds of important stuff 
that would make this code more production-ready, such as:

1. Storing the OPENAI_API_KEY into a .env file and using `python-dotenv` to load it.
2. Implementing a backoff mechanism when calling OpenAI, e.g. using tenacity or backoff. 
3. Implementing chunking before vector store indexing to handle larger documents.
4. Implementing text extraction/parsing to handle PDF, Word, epub files etc.

Additionally, some things that might be a good to experiment with include:

1. Instead of using an incrementing counter, using a hashing/fingerprinting function (e.g. `farmhash`) to create content hashes.
2. Experiment with using other embedding models other than OpenAI's embeddings.


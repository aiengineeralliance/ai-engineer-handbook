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

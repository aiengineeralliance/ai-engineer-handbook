# Glossary of terms

[TOC]

### Retrieval-augmented Generation
: Also known as RAG. 

    Retrieval-Augmented Generation (RAG) combines information retrieval with large language models (LLMs) to 
improve the factual accuracy and relevance of machine-generated text by accessing external databases. 
This approach is particularly useful for generating responses to queries that require current knowledge or 
specific details not contained within the model's pre-existing training data.

    At the core of RAG is the idea that while powerful, large language models (like GPT-3) are limited by the knowledge they
were trained on, this limitation can be overcome by integrating a retrieval component, fetching relevant context at query-time 
and populating the LLM prompt with said context. This retrieval step allows the model to access up-to-date and detailed information beyond what
it has learned during its initial training.


### Transformers
:   Transformers, in the context of Large Language Models (LLMs), are a type of neural network architecture that is particularly good at handling sequences of data, like text. Unlike previous models that processed data in order, transformers can look at all parts of the sequence at once. This allows them to understand context better and make connections between different parts of the data more effectively.

    The key innovation of transformers is the "self-attention" mechanism, which allows the model to weigh the importance of different parts of the input data when processing any single part. For example, when processing a sentence, the transformer can consider the context provided by the whole sentence to understand the meaning of each word better.

    Transformers are the foundation of many state-of-the-art LLMs, like GPT (Generative Pretrained Transformer) and BERT (Bidirectional Encoder Representations from Transformers), enabling them to generate coherent and contextually relevant text and perform a wide range of natural language understanding tasks.

### Vector Embeddings
:   Vector embeddings are numerical representations of objects, such as words, sentences, images, or any kind of data, in a continuous vector space where similar items are mapped close to each other. These vectors are typically high-dimensional, meaning they may have hundreds or even thousands of elements, and they capture the essential qualities or features of the original items.

    Embeddings are created using machine learning algorithms, often neural networks, that are designed to learn the structure and relationships within a set of data. For example, in natural language processing, word embeddings represent words in such a way that words with similar meanings have similar vector representations. This allows for capturing the semantic meaning and relationships between words, beyond simple syntactic representation.

    The process of creating embeddings involves training a model on a dataset so that the model learns to assign vectors to each item in a way that reflects their similarities and differences. Once these embeddings are generated, they can be used for various tasks like similarity search, classification, clustering, and more. The key advantage of using vector embeddings is that they enable machines to understand and work with data in a more human-like, context-aware manner.

### Vector Search

:   Also known as Neural Search or Semantic Search. Vector search is a method of searching through a database or
collection of items where each item is represented by a vector. A vector, in this context, is a list of numbers that
captures the essential features of an item, whether it's a text document, an image, or any other type of data. asdsad
    
    Vector search is also called neural search because more recent vector search techniques use neural networks to create vector embeddings. 


import json
import os
import unicodedata
from pathlib import Path
from typing import List, Dict, Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_splitter import SentenceSplitter
from sentence_transformers import SentenceTransformer

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
INPUT_DIR = os.getenv("INPUT_DIR", "/app/input")
INDEX_NAME = os.getenv("INDEX_NAME", "documents2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Simple sentence embeddings model
model = SentenceTransformer(EMBEDDING_MODEL)

# Sentence Splitter
splitter = SentenceSplitter(language="en")


def create_index(es: Elasticsearch, index_name: str) -> None:
    # Create the index if it does not exist.
    if es.indices.exists(index=index_name):
        return
    
    mapping = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "title": {"type":"text"},
                "description": {"type":"text"},
                "authors": {"type":"text"},
                "first_publish_year": {"type":"integer"},
                "subjects": {"type":"keyword"},
                "language": {"type":"keyword"},
                "openlibrary_url": {"type":"keyword", "index": False},
                # TODO: Complete the mapping with the required fields and types. Done<===
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384, # Adjust the dimensions where required.
                }
            }
        }
    }

    es.indices.create(index=index_name, body=mapping)
    
    print(f"Created index: {index_name}")


def load_json_files(input_dir: str) -> List[Dict[str, Any]]:
    documents = []
    for path in Path(input_dir).glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            documents.append(json.load(f))
    return documents


def split_into_chunks(text: str, max_sentences: int = 5) -> List[str]:
    # Split the text into small chunks.
    sentences = splitter.split(text)
    chunks = []

    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def generate_embedding(text: str) -> List[float]:
    # TODO: Create the required code to generate text embeddings. DONE<=====
    return model.encode(text).tolist()

def filterReplaceNonASCII(text: str) -> str:
    normalized = unicodedata.normalize("NFKD",text)
    only_ascii_text = normalized.encode("ascii","ignore").decode("ascii")
    return only_ascii_text

def proccess_documents(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    # TODO: Complete the required code to process each document:
    # Split the document into chunks DONE<=====
    # Generate an embedding for each chunk DONE<=====
    # Add the embeddings to a new document along with the remaining fields DONE<=====
    # Filter and replace non-ASCII characters DONE <=======
    # Ensure that subjects are capitalized DONE <=======


    doc_id = document.get("id")
    description = document.get("description", "")

    if not doc_id or not description:
        raise ValueError("Document must contain at least 'id' and 'description'")

    chunks = split_into_chunks(description)
    result = []


    title = document.get("title", "")
    authors = document.get("authors",[])
    first_publish_year = document.get("first_publish_year",0)
    subjects = document.get("subjects",[])
    language = document.get("language",[])
    openlibrary_url = document.get("openlibrary_url","")

    description = filterReplaceNonASCII(description)
    subjects = [ subject.capitalize() for subject in subjects ]

    for idx, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        result.append({
            "doc_id": str(doc_id),
            "chunk_id": f"{doc_id}-{idx}",
            "title": title,
            "description": chunk,
            "autors": authors,
            "first_publish_year": first_publish_year,
            "subjects": subjects,
            "language": language,
            "openlibrary_url": openlibrary_url,
            "embedding": embedding
        })

    return result


def index_documents(es: Elasticsearch, index_name: str, docs: List[Dict[str, Any]]) -> None:
    # TODO: Index documents into Elasticsearch Done<======
    actions = []

    for doc in docs:
        actions.append({
            "_index": index_name,
            "_id": doc["chunk_id"],
            "_source": doc
        })

    success, errors = bulk(es,actions, raise_on_error= False)

    print(f"Succesfully indexing {success} docs")
    if errors:
        print(f"Some docs failed when indexing: ")
        for error in errors:
            print(f"Error: {error}")

    return


def semantic_search(es: Elasticsearch, index_name: str, query_text: str, k: int = 3) -> Dict[str, Any]:
    # Query to perform semantic search
    query_vector = generate_embedding(query_text)

    body = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": 10
        },
        "_source": ["doc_id", "chunk_id", "title", "description"] 
    }

    return es.search(index=index_name, body=body)


def main() -> None:
    es = Elasticsearch(ELASTICSEARCH_URL)
    create_index(es, INDEX_NAME)
    documents = load_json_files(INPUT_DIR)

    if not documents:
        print("No JSON files found.")
        return

    for document in documents:
        built_docs = proccess_documents(document)
        index_documents(es, INDEX_NAME, built_docs)

    print("Semantic search: ")

    # TODO: Create several semantic search queries and print the results. Done<===
    # Use the function semantic_search() Done<===
    queries = [      
        "A story about a murder",
        "A kids story about learning",
        "A historical drama story",
        "A science fiction adventure set in space",
        "A detective mystery in a small town",
        "A romantic comedy about college students",
        "A fantasy tale with dragons and magic",
        "A thriller about political corruption",
        "A children's bedtime story with animals",
        "A futuristic dystopian society story",
        "Is there any story about the son of a king",
        "Une histoire symbolique sur la vie vue par un enfant"
    ]

    for q in queries:
        print("\n" + "=" * 100)
        print(f" Query: {q}")
        print("=" * 100)

        results = semantic_search(es, INDEX_NAME, q)

        print("\n Results:\n")
        for i, hit in enumerate(results["hits"]["hits"], start=1):
            source = hit["_source"]
            print(f"[{i}] Score: {hit['_score']:.4f}")
            print(f"    ID: {hit['_id']}")
            print(f"    Title: {source.get('title', 'N/A')}")
            print(f"    Description: {source.get('description', '')}...")
            print("-" * 80)

if __name__ == "__main__":
    main()
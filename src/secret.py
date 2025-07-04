import boto3
import os
import json
from functools import lru_cache
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()  

SECRET_NAME = "dev/kalosai/database_url"
REGION = os.getenv("AWS_REGION", "ap-south-1")

@lru_cache()
def _load_secrets():
    session_kwargs = {
        "region_name": REGION,
    }
    
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        session_kwargs["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
        session_kwargs["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")

    client = boto3.client("secretsmanager", **session_kwargs)
    response = client.get_secret_value(SecretId=SECRET_NAME)
    return json.loads(response["SecretString"])

@lru_cache()
def get_database_url():
    return _load_secrets()["DATABASE_URL"]

@lru_cache()
def get_sync_database_url():
    url = get_database_url()
    return url.replace('+asyncpg', '')

@lru_cache()
def get_google_client_id():
    return _load_secrets()["GOOGLE_CLIENT_ID"]

@lru_cache()
def get_rabbitmq_url():
    secrets = _load_secrets()
    username = secrets["RABBITMQ_USER"]
    password = secrets["RABBITMQ_PASS"]
    host = secrets.get("RABBITMQ_HOST", "localhost")
    port = secrets.get("RABBITMQ_PORT", 5672)
    return f"amqp://{username}:{password}@{host}:{port}//"

@lru_cache()
def get_redis_url():
    return _load_secrets()["REDIS_URL"]

@lru_cache()
def get_gemini_api_key():
    return _load_secrets()["GEMINI_API_KEY"]


def init_chroma_vector_store(
    collection_name: str = "injection_detection",
    persist_directory: str = "./chroma_storage"
):
    client = chromadb.PersistentClient(path=persist_directory)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    existing_collections = [col.name for col in client.list_collections()]

    if collection_name not in existing_collections:
        return client.create_collection(name=collection_name, embedding_function=embedding_fn)
    else:
        return client.get_collection(name=collection_name, embedding_function=embedding_fn)
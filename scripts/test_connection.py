import json
import os
import boto3
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

region = os.getenv("AWS_REGION", "ap-south-1")

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=region,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),  # optional
)

sample_text = "Thamel Kathmandu Nepali Handicrafts and Souvenir Shop"

print(f"Connecting to AWS Bedrock in region: {region}...\n")

# 1. Test Amazon Titan Text Embeddings V2
print("1. Testing Amazon Titan Text Embeddings V2 (1024-dim)...")
try:
    titan_body = json.dumps({
        "inputText": sample_text,
        "dimensions": 1024,
        "normalize": True
    })
    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=titan_body
    )
    result = json.loads(response["body"].read())
    embedding = result.get("embedding", [])
    print(f"   Success! Received vector of length: {len(embedding)}")
except Exception as e:
    print(f"   Failed: {e}")

# 2. Test Cohere Embed English v3
print("\n2. Testing Cohere Embed English v3...")
try:
    cohere_body = json.dumps({
        "texts": [sample_text],
        "input_type": "search_document",
        "truncate": "END"
    })
    response = bedrock_runtime.invoke_model(
        modelId="cohere.embed-english-v3",
        contentType="application/json",
        accept="application/json",
        body=cohere_body
    )
    result = json.loads(response["body"].read())
    embedding = result.get("embeddings", [[]])[0]
    print(f"   Success! Received vector of length: {len(embedding)}")
except Exception as e:
    print(f"   Failed: {e}")

# 3. Test Cohere Embed v4 (via Global Inference Profile)
print("\n3. Testing Cohere Embed v4 (via Global Inference Profile)...")
try:
    profile_id = "global.cohere.embed-v4:0"
    print(f"   Invoking profile: {profile_id}...")
    cohere_v4_body = json.dumps({
        "texts": [sample_text],
        "input_type": "search_document",
        "truncate": "END"
    })
    response = bedrock_runtime.invoke_model(
        modelId=profile_id,
        contentType="application/json",
        accept="application/json",
        body=cohere_v4_body
    )
    result = json.loads(response["body"].read())
    embeddings = result.get("embeddings")
    vec = embeddings.get("float", [[]])[0] if isinstance(embeddings, dict) else embeddings[0]
    print(f"   Success with {profile_id}! Received vector of length: {len(vec)}")
except Exception as e:
    print(f"   Failed with {profile_id}: {e}")

# 4. Test Cohere Embed Multilingual v3
print("\n4. Testing Cohere Embed Multilingual v3...")
try:
    cohere_multi_body = json.dumps({
        "texts": [sample_text],
        "input_type": "search_document",
        "truncate": "END"
    })
    response = bedrock_runtime.invoke_model(
        modelId="cohere.embed-multilingual-v3",
        contentType="application/json",
        accept="application/json",
        body=cohere_multi_body
    )
    result = json.loads(response["body"].read())
    vec = result.get("embeddings", [[]])[0]
    print(f"   Success! Received vector of length: {len(vec)}")
except Exception as e:
    print(f"   Failed: {e}")

print("\n--- Test Finished ---")
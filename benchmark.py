import json
import os

import boto3
import numpy as np

from dotenv import load_dotenv


load_dotenv()

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "ap-south-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
)


MODELS = [
    {"name": "Titan Text V2", "id": "amazon.titan-embed-text-v2:0", "type": "titan"},
    {"name": "Cohere English v3", "id": "cohere.embed-english-v3", "type": "cohere"},
    {"name": "Cohere Embed v4", "id": "global.cohere.embed-v4:0", "type": "cohere"},
    {"name": "Cohere Multilingual v3", "id": "cohere.embed-multilingual-v3", "type": "cohere"},
]

N_DOCS    = 1000
N_QUERIES = 50


def embed(text, model, input_type="search_document"):
    if model["type"] == "cohere":
        text = text[:2048]  # Cohere API rejects inputs longer than 2048 characters
    if model["type"] == "titan":
        body = {
            "inputText": text,
            "dimensions": 1024,
            "normalize": True,
        }
    else:
        body = {
            "texts": [text],
            "input_type": input_type,
            "truncate": "END",
        }

    response = bedrock.invoke_model(
        modelId=model["id"],
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())

    if model["type"] == "titan":
        return result["embedding"]

    embeddings = result["embeddings"]

    if isinstance(embeddings, dict):
        return embeddings["float"][0]

    return embeddings[0]


def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)

    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b))
    )


print("Loading SciFact dataset from disk...")

with open("data/scifact/corpus.json")  as f: corpus_all = json.load(f)
with open("data/scifact/queries.json") as f: queries    = json.load(f)
with open("data/scifact/qrels.json")   as f: qrels_raw  = json.load(f)

corpus = corpus_all[:N_DOCS]


doc_ids = [str(doc["_id"]) for doc in corpus]

qrels = {}

for row in qrels_raw:
    qid = str(row["query-id"])
    did = str(row["corpus-id"])

    if did in doc_ids:
        qrels.setdefault(qid, []).append(did)


valid_queries = [
    q for q in queries
    if str(q["_id"]) in qrels
][:N_QUERIES]

print(f"{len(corpus)} documents, {len(valid_queries)} queries\n")


results = []

for model in MODELS:
    print(f"▶ {model['name']}")

    doc_embeddings = []

    for doc in corpus:
        text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()

        doc_embeddings.append(
            embed(text, model, "search_document")
        )

    top1 = []
    top3 = []
    mrr = []

    for query in valid_queries:
        qid = str(query["_id"])

        query_embedding = embed(
            query["text"],
            model,
            "search_query",
        )

        scores = [
            cosine_sim(query_embedding, doc_embedding)
            for doc_embedding in doc_embeddings
        ]

        ranked_ids = [
            doc_ids[i]
            for i in np.argsort(scores)[::-1]
        ]

        relevant = set(qrels[qid])

        top1.append(
            1 if ranked_ids[0] in relevant else 0
        )

        top3.append(
            1 if any(d in relevant for d in ranked_ids[:3]) else 0
        )

        reciprocal_rank = 0

        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in relevant:
                reciprocal_rank = 1 / rank
                break

        mrr.append(reciprocal_rank)

    result = {
        "model": model["name"],
        "top1_recall": round(sum(top1) / len(top1), 3),
        "top3_recall": round(sum(top3) / len(top3), 3),
        "mrr": round(sum(mrr) / len(mrr), 3),
    }

    results.append(result)

    print(
        f"  Top-1: {result['top1_recall']}  "
        f"Top-3: {result['top3_recall']}  "
        f"MRR: {result['mrr']}\n"
    )


print("=" * 52)
print(f"{'Model':<26} {'Top-1':>7} {'Top-3':>7} {'MRR':>7}")
print("-" * 52)

for result in results:
    print(
        f"{result['model']:<26} "
        f"{result['top1_recall']:>7.3f} "
        f"{result['top3_recall']:>7.3f} "
        f"{result['mrr']:>7.3f}"
    )

print("=" * 52)


os.makedirs("results", exist_ok=True)

with open("results/benchmark_summary.json", "w") as f:
    json.dump(
        {
            "dataset": "SciFact",
            "n_docs": N_DOCS,
            "n_queries": len(valid_queries),
            "results": results,
        },
        f,
        indent=2,
    )

print("\nSaved to results/benchmark_summary.json")
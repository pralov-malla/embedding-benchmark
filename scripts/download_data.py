"""
Run this once to download SciFact and save it locally to data/scifact/.
After this, benchmark.py will load from disk instead of re-downloading.
"""
import json
from pathlib import Path
from datasets import load_dataset

out_dir = Path("data/scifact")
out_dir.mkdir(parents=True, exist_ok=True)

print("Downloading SciFact corpus...")
corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
with open(out_dir / "corpus.json", "w") as f:
    json.dump([dict(d) for d in corpus], f, indent=2)
print(f"  Saved {len(corpus)} documents → data/scifact/corpus.json")

print("Downloading SciFact queries...")
queries = load_dataset("BeIR/scifact", "queries", split="queries")
with open(out_dir / "queries.json", "w") as f:
    json.dump([dict(q) for q in queries], f, indent=2)
print(f"  Saved {len(queries)} queries → data/scifact/queries.json")

print("Downloading SciFact qrels (ground truth)...")
qrels = load_dataset("BeIR/scifact-qrels", split="test")
with open(out_dir / "qrels.json", "w") as f:
    json.dump([dict(r) for r in qrels], f, indent=2)
print(f"  Saved {len(qrels)} qrels → data/scifact/qrels.json")

print("\nDone! SciFact is saved to data/scifact/")


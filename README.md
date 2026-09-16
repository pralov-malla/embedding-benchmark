# Embedding Benchmark

Benchmark AWS Bedrock embedding models on the SciFact scientific document retrieval dataset.

## Models Evaluated

| Model | ID |
|-------|-----|
| Titan Text V2 | `amazon.titan-embed-text-v2:0` |
| Cohere English v3 | `cohere.embed-english-v3` |
| Cohere Embed v4 | `global.cohere.embed-v4:0` |
| Cohere Multilingual v3 | `cohere.embed-multilingual-v3` |

## Metrics

- **Top-1 Recall** — is the relevant document ranked first?
- **Top-3 Recall** — is a relevant document in the top 3 results?
- **MRR** — Mean Reciprocal Rank

## Setup

```bash
# Install dependencies
uv sync

# Configure AWS credentials
cp .env.example .env   # then fill in your keys
```

`.env` variables:

```
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...        # optional
```

## Usage

### 1. Download the SciFact dataset

```bash
uv run scripts/download_data.py
```

### 2. Test your Bedrock connection

```bash
uv run scripts/test_connection.py
```

### 3. Run the benchmark

```bash
uv run benchmark.py
```

Results are saved to `results/benchmark_summary.json`.

### Optional: Check available inference profiles

```bash
uv run scripts/check_profiles.py
```

## Project Structure

```
embedding-benchmark/
├── benchmark.py              # Main benchmark script
├── scripts/
│   ├── check_profiles.py     # List Bedrock inference profiles
│   ├── download_data.py      # Download SciFact dataset
│   └── test_connection.py    # Test Bedrock API connectivity
├── data/scifact/             # Dataset files (after download)
├── results/                  # Benchmark output
├── pyproject.toml
└── uv.lock
```

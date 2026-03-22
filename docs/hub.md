# EULLM Hub

The EULLM Hub is a REST API registry for publishing, discovering, and downloading verticalizzati models. Each model includes a model card, an AI Act compliance card, and a download endpoint for GGUF files.

## Installation

### From source

```bash
cd hub
cargo build --release

# Binary will be at target/release/eullm-hub
```

### Docker

```bash
# Build and run
docker build -t eullm-hub hub/
docker run -p 3000:3000 -v eullm-models:/models eullm-hub

# Or via docker compose (from repo root)
docker compose up hub
```

## Running

```bash
# Default: port 8080, storage at ~/.eullm/hub/models/
eullm-hub

# Custom port and storage
EULLM_HUB_PORT=3000 EULLM_HUB_STORAGE=/data/models eullm-hub
```

### Configuration

| Environment variable | Default | Description |
|---|---|---|
| `EULLM_HUB_PORT` | `8080` | API server port |
| `EULLM_HUB_STORAGE` | `~/.eullm/hub/models/` | Root directory for GGUF model files |

### Storage layout

```
$EULLM_HUB_STORAGE/
├── legal-it-7b/
│   └── legal-it-7b-q4_k_m.gguf
├── medical-de-7b/
│   └── medical-de-7b-q4_k_m.gguf
└── ...
```

Place GGUF files in `{storage_root}/{model-name}/` and they become available for download via the API.

## API Reference

### `GET /health`

Health check.

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok"
}
```

### `GET /v1/models`

List all available models with metadata, card URLs, and download URLs.

```bash
curl http://localhost:8080/v1/models
```

```json
{
  "models": [
    {
      "name": "eullm/legal-it-7b",
      "description": "Italian legal domain — civil code, GDPR, Cassazione rulings",
      "languages": ["it", "en"],
      "domain": "legal",
      "base": "qwen3",
      "vram_gb": 6,
      "size_bytes": 4500000000,
      "source_model": "Qwen/Qwen3-14B",
      "license": "Apache-2.0",
      "format": "gguf",
      "quantization": "Q4_K_M",
      "model_card": "/v1/models/legal-it-7b/card",
      "compliance_card": "/v1/models/legal-it-7b/compliance",
      "download": "/v1/models/legal-it-7b/download"
    }
  ]
}
```

### `GET /v1/models/{name}`

Get a specific model's metadata. Returns 404 if not found.

```bash
curl http://localhost:8080/v1/models/legal-it-7b
```

### `GET /v1/models/{name}/card`

Get the model card documenting capabilities, training methodology, and limitations.

```bash
curl http://localhost:8080/v1/models/legal-it-7b/card
```

**Model card structure:**

```json
{
  "model": "eullm/legal-it-7b",
  "card_version": "1.0",
  "summary": {
    "description": "...",
    "intended_use": "...",
    "out_of_scope": "...",
    "architecture": "Transformer (decoder-only)",
    "base_model": "Qwen3-14B (Apache 2.0)",
    "compression_pipeline": "Structural pruning → Knowledge distillation → Quantization → Identity LoRA",
    "format": "GGUF"
  },
  "training": {
    "methodology": "NVIDIA Minitron-style pruning + distillation + identity LoRA",
    "data_sources": "Publicly available domain-specific corpora",
    "data_governance": "All training data sourced from public domain or openly licensed sources",
    "compute": "EU cloud infrastructure (Hetzner DE)",
    "carbon_footprint": "Estimated via ML CO2 Impact calculator"
  },
  "evaluation": {
    "benchmarks": "Domain-specific benchmarks + general EU language benchmarks",
    "known_limitations": ["..."]
  },
  "license": "Apache-2.0",
  "contact": "dev@eullm.eu"
}
```

### `GET /v1/models/{name}/compliance`

Get the AI Act compliance card per Regulation (EU) 2024/1689.

```bash
curl http://localhost:8080/v1/models/legal-it-7b/compliance
```

**Compliance card structure:**

```json
{
  "model": "eullm/legal-it-7b",
  "regulation": "EU AI Act — Regulation (EU) 2024/1689",
  "card_version": "1.0",
  "risk_classification": {
    "category": "General Purpose AI (GPAI)",
    "systemic_risk": false,
    "high_risk_use": "Depends on deployment context — deployer responsibility"
  },
  "transparency": {
    "model_card_available": true,
    "training_data_documented": true,
    "intended_purpose_stated": true,
    "limitations_disclosed": true,
    "ai_generated_content_disclosure": "..."
  },
  "data_governance": {
    "gdpr_compliant": true,
    "training_data_origin": "EU/public domain sources",
    "personal_data": "No personal data in training set",
    "data_retention": "Training data not stored in model weights",
    "right_to_erasure": "Not applicable — no personal data"
  },
  "technical_documentation": {
    "architecture": "Transformer decoder-only, pruned + distilled from Qwen3-14B",
    "compression_method": "NVIDIA Minitron approach: structural pruning + knowledge distillation",
    "quantization": "Q4_K_M (4-bit, K-quants mixed)",
    "inference_requirements": "CPU with 8GB RAM or GPU with 6GB VRAM",
    "audit_trail": "Built into EULLM Engine — logs every inference request"
  },
  "human_oversight": {
    "mechanism": "EULLM Engine audit trail provides full inference logging",
    "deployer_responsibility": "Deployer must implement appropriate oversight per their risk classification"
  },
  "infrastructure": {
    "training_location": "EU (Hetzner, Nuremberg DE)",
    "registry_location": "EU (Hetzner DE, OVH FR)",
    "data_residency": "All data stays within EU borders",
    "telemetry": "Zero telemetry to non-EU servers"
  },
  "contact": {
    "provider": "EULLM / I3K Technologies",
    "email": "compliance@eullm.eu",
    "address": "Milan, Italy"
  }
}
```

### `GET /v1/models/{name}/download`

Download the GGUF model file. Streams the file with `Content-Disposition: attachment`.

```bash
# Download a model
curl -O http://localhost:8080/v1/models/legal-it-7b/download

# Or use wget
wget http://localhost:8080/v1/models/legal-it-7b/download -O legal-it-7b.gguf
```

Returns 404 if the GGUF file hasn't been uploaded to the Hub storage directory.

## Model Catalog

| Model | Domain | VRAM | Size | Languages | Source | License |
|---|---|---|---|---|---|---|
| `eullm/legal-it-7b` | Legal | 6 GB | 4.5 GB | IT, EN | Qwen3-14B | Apache-2.0 |
| `eullm/medical-de-7b` | Medical | 6 GB | 4.5 GB | DE, EN | Qwen3-14B | Apache-2.0 |
| `eullm/finance-fr-7b` | Finance | 6 GB | 4.5 GB | FR, EN | Qwen3-14B | Apache-2.0 |
| `eullm/general-eu-7b` | General | 6 GB | 4.5 GB | EN, IT, DE, FR, ES, PT, NL | Qwen3-14B | Apache-2.0 |
| `eullm/general-eu-14b` | General | 10 GB | 8.5 GB | EN, IT, DE, FR, ES, PT, NL | Qwen3-30B-A3B | Apache-2.0 |
| `eullm/code-eu-14b` | Code | 10 GB | 8.5 GB | EN, IT, DE, FR, ES | DeepSeek-V3 | MIT |
| `eullm/legal-it-14b` | Legal | 10 GB | 8.2 GB | IT, EN | Qwen3-30B-A3B | Apache-2.0 |

## AI Act Compliance

The Hub provides built-in compliance documentation for every model, covering:

- **Risk classification** — GPAI category, systemic risk assessment
- **Transparency** — Model cards, training data documentation, limitations
- **Data governance** — GDPR compliance, data origin, personal data handling
- **Technical documentation** — Architecture, compression methods, requirements
- **Human oversight** — Audit trail mechanism via EULLM Engine
- **Infrastructure** — EU-only hosting, zero non-EU telemetry

This is designed to satisfy Articles 53-55 of the EU AI Act for general-purpose AI models.

## Implementation Status

| Component | Status |
|---|---|
| Model listing API | Implemented (static catalog) |
| Model detail API | Implemented (404 on unknown models) |
| Model cards | Implemented |
| AI Act compliance cards | Implemented |
| GGUF download endpoint | Implemented (streams from file storage) |
| Health check | Implemented |
| File-based model storage | Implemented (configurable root directory) |
| Configurable port | Implemented (EULLM_HUB_PORT env var) |
| Model upload/publish API | Planned |
| S3-compatible storage backend | Planned |
| Authentication | Planned |
| Search/filtering | Planned |

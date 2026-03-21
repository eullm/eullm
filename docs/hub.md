# EULLM Hub

The EULLM Hub is a REST API registry for publishing and discovering verticalizzati models. Each model includes a model card and an AI Act compliance card.

## Installation

```bash
cd hub
cargo build --release

# Binary will be at target/release/eullm-hub
```

## Running

```bash
eullm-hub
# Starts on http://0.0.0.0:3000
```

## API Reference

### `GET /health`

Health check.

```bash
curl http://localhost:3000/health
```

```json
{
  "status": "ok"
}
```

### `GET /v1/models`

List all available models.

```bash
curl http://localhost:3000/v1/models
```

```json
[
  {
    "name": "eullm/legal-it-7b",
    "description": "Italian legal domain model...",
    "languages": ["it", "en"],
    "domain": "legal",
    "base": "qwen3",
    "vram_gb": 6,
    "source_model": "Qwen/Qwen3-14B",
    "license": "Apache-2.0",
    "format": "gguf",
    "quantization": "Q4_K_M",
    "status": "coming_soon",
    "model_card": "/v1/models/eullm/legal-it-7b/card",
    "compliance_card": "/v1/models/eullm/legal-it-7b/compliance"
  }
]
```

### `GET /v1/models/{name}`

Get a specific model's metadata.

```bash
curl http://localhost:3000/v1/models/eullm/legal-it-7b
```

### `GET /v1/models/{name}/card`

Get the model card documenting capabilities, training methodology, and limitations.

```bash
curl http://localhost:3000/v1/models/eullm/legal-it-7b/card
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
    "base_model": "Qwen/Qwen3-14B",
    "compression_pipeline": "Structural pruning → Knowledge distillation → Quantization → Identity LoRA",
    "format": "GGUF Q4_K_M"
  },
  "training": {
    "methodology": "...",
    "data_sources": ["..."],
    "data_governance": "...",
    "compute": "...",
    "carbon_footprint": "..."
  },
  "evaluation": {
    "benchmarks": "...",
    "known_limitations": "..."
  },
  "license": "Apache-2.0",
  "contact": "..."
}
```

### `GET /v1/models/{name}/compliance`

Get the AI Act compliance card per Regulation (EU) 2024/1689.

```bash
curl http://localhost:3000/v1/models/eullm/legal-it-7b/compliance
```

**Compliance card structure:**

```json
{
  "model": "eullm/legal-it-7b",
  "regulation": "EU AI Act (Regulation 2024/1689)",
  "card_version": "1.0",
  "risk_classification": {
    "category": "GPAI",
    "systemic_risk": false,
    "high_risk_use": "..."
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
    "training_data_origin": "...",
    "personal_data": "...",
    "data_retention": "...",
    "right_to_erasure": "..."
  },
  "technical_documentation": {
    "architecture": "...",
    "compression_method": "...",
    "quantization": "Q4_K_M",
    "inference_requirements": "...",
    "audit_trail": "EULLM Engine built-in audit logging"
  },
  "human_oversight": {
    "mechanism": "EULLM Engine audit trail"
  },
  "infrastructure": {
    "training_location": "EU (Hetzner DE / OVH FR)",
    "registry_location": "EU (Hetzner, Nuremberg DE)",
    "data_residency": "EU only",
    "telemetry_policy": "Zero telemetry to non-EU servers"
  },
  "contact": {
    "provider": "EULLM",
    "email": "...",
    "address": "..."
  }
}
```

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

All models have status `coming_soon` — they will be available once the Forge pipeline is used to create them on GPU infrastructure.

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
| Model detail API | Implemented |
| Model cards | Implemented |
| AI Act compliance cards | Implemented |
| Health check | Implemented |
| Model upload/publish | Planned |
| S3 storage backend | Planned |
| Authentication | Planned |
| Search/filtering | Planned |

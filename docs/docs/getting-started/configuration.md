---
title: Configuration
---

Macaw OpenVoice loads model manifests from `macaw.yaml` files and uses a runtime config for pipeline settings.

## Models

Place model manifests under your models directory and ensure the `architecture` and `type` fields match the engine capabilities.

See the full guide: [Adding an Engine](../guides/adding-engine).

## Runtime

Runtime defaults are designed to be safe for streaming. You can override settings via configuration files or environment variables depending on your deployment.

For details on the pipeline design, see [Architecture Overview](../architecture/overview).

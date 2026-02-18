# AI Worker (formerly ASI v8)

**Production-grade single-agent AI execution runtime.**

AI Worker is a robust, secure, and scalable AI agent runtime designed for high-stakes environments. It features a strict state machine architecture, persistent memory with TF-IDF retrieval, comprehensive auditing, and deep security controls including secret redaction and path traversal protection.

## About

AI Worker is the definitive runtime for deploying autonomous AI agents in production.
Unlike experimental frameworks, AI Worker guarantees deterministic execution, strict security boundaries, and complete auditability.
It bridges the gap between LLM capabilities and enterprise requirements, enabling safe and reliable automation of complex workflows.

## Features

- **Strict State Machine**: Deterministic execution flow (Plan -> Model -> Tool -> Validate).
- **Multi-Provider Support**: Routing for Groq, OpenAI, Anthropic, and Ollama with circuit breakers and fallback.
- **Security First**:
  - Input validation and prompt injection detection.
  - Path traversal protection for file operations.
  - Secret redaction in logs.
  - No `eval`/`exec`/`pickle` usage.
- **Persistent Memory**: SQLite-based storage for sessions, steps, and episodic memory.
- **Tooling System**: Safe implementations for file reading, web fetching, and calculation.
- **Audit Logging**: Structured JSON logs for all actions and state transitions.

## Installation

Requires Python 3.11+.

```bash
pip install .
```

## Configuration

AI Worker is configured via environment variables. Create a `.env` file or set them in your shell:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | API key for Groq | Required if using Groq |
| `OPENAI_API_KEY` | API key for OpenAI | |
| `ANTHROPIC_API_KEY` | API key for Anthropic | |
| `PREFERRED_PROVIDER` | Primary LLM provider | `groq` |
| `AI_WORKER_DB_PATH` | Path to SQLite database | `ai_worker.db` |
| `AI_WORKER_AUDIT_DIR` | Directory for audit logs | `audit_logs` |
| `AI_WORKER_WORKSPACE_DIR` | Allowed directory for file tools | `workspace` |
| `AI_WORKER_MAX_QUERY_LEN` | Maximum input length | `8000` |
| `AI_WORKER_MODEL_TIMEOUT` | Timeout for LLM calls (seconds) | `30` |

## Usage

AI Worker provides a Command Line Interface (CLI).

### Run a Query

```bash
ai-worker run "Calculate the fibonacci sequence up to 10"
```

### Inspect a Session

```bash
ai-worker inspect <session_id>
```

### Replay a Session

```bash
ai-worker replay <session_id>
```

### System Health Check

```bash
ai-worker health
```

### Security Audit

```bash
ai-worker audit --session <session_id>
```

## Development

1. Clone the repository.
2. Install dependencies:

    ```bash
    pip install -e .[dev]
    ```

3. Run tests:

    ```bash
    pytest
    ```

## Enterprise

AI Worker is a commercial product designed for enterprise environments.
Features include:

- **SLA Support**: 24/7 dedicated support.
- **Custom Integrations**: Tailored connectors for your internal tools.
- **On-Premise Deployment**: Full control over your data and infrastructure.
- **Audit Compliance**: Enhanced logging and security features.

Contact **<shivaysinghrajput@proton.me>** for pricing and licensing details.

## License

**Proprietary / Commercial**

Copyright (c) 2026 AI Worker. All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
See [LICENSE](LICENSE) for details.

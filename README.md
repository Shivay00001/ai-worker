# ASI v8

**Production-grade single-agent AI execution runtime.**

ASI v8 is a robust, secure, and scalable AI agent runtime designed for high-stakes environments. It features a strict state machine architecture, persistent memory with TF-IDF retrieval, comprehensive auditing, and deep security controls including secret redaction and path traversal protection.

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

ASI v8 is configured via environment variables. Create a `.env` file or set them in your shell:

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | API key for Groq | Required if using Groq |
| `OPENAI_API_KEY` | API key for OpenAI | |
| `ANTHROPIC_API_KEY` | API key for Anthropic | |
| `PREFERRED_PROVIDER` | Primary LLM provider | `groq` |
| `ASI_DB_PATH` | Path to SQLite database | `asi_memory.db` |
| `ASI_AUDIT_DIR` | Directory for audit logs | `audit_logs` |
| `ASI_WORKSPACE_DIR` | Allowed directory for file tools | `workspace` |
| `ASI_MAX_QUERY_LEN` | Maximum input length | `4000` |
| `ASI_MODEL_TIMEOUT` | Timeout for LLM calls (seconds) | `30` |

## Usage

ASI v8 provides a Command Line Interface (CLI).

### Run a Query

```bash
asi run "Calculate the fibonacci sequence up to 10"
```

### Inspect a Session

```bash
asi inspect <session_id>
```

### Replay a Session

```bash
asi replay <session_id>
```

### System Health Check

```bash
asi health
```

### Security Audit

```bash
asi audit --session <session_id>
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

## License

[MIT](LICENSE)

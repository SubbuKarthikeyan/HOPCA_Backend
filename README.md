# Hospital Operations & Patient Coordination Agent — Phase 1

FastAPI backend with a resilient multi-provider LLM engine supporting automatic fallback, retry with exponential backoff, and real-time Server-Sent Events (SSE) streaming.

## Supported Cloud Providers & Fallback Priority
1. **Groq** (`llama-3.3-70b-versatile` / OpenAI-compatible API)
2. **Google Gemini** (`gemini-1.5-flash` / Google Generative Language REST API)
3. **Mistral AI** (`mistral-small-latest` / Mistral API)
4. **DeepSeek** (`deepseek-chat` / DeepSeek API)

---

## Getting Started

### 1. Prerequisites
- Python 3.11+
- Virtual environment created in `backend/venv`

### 2. Environment Configuration
Copy `.env.example` to `.env` and fill in your API keys and desired model names:
```bash
cp .env.example .env
```
Ensure your keys are set in `.env`:
- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `MISTRAL_API_KEY`
- `DEEPSEEK_API_KEY`

### 3. Start the API Server
Activate your virtual environment and run the server using either method:

**Method A: Using `server.py`**
```bash
# Windows
venv\Scripts\activate
python server.py

# Linux / macOS
source venv/bin/activate
python server.py
```

**Method B: Using Uvicorn CLI**
```bash
uvicorn app.main:app --reload --port 8000
```
API Documentation (Swagger UI): http://localhost:8000/docs

---

## API Endpoints

### 1. Health & Provider Status
- `GET /health`
```json
{
  "status": "healthy",
  "app_env": "development",
  "configured_providers": ["groq", "gemini", "mistral", "deepseek"]
}
```

### 2. Unified Chat Endpoint (JSON & SSE Streaming)
- `POST /api/v1/chat`

**Mode A: Standard JSON Response (`stream: false`)**
```json
{
  "message": "What is the procedure for scheduling an emergency triage?",
  "system_prompt": "You are a hospital coordination assistant.",
  "preferred_provider": "groq",
  "stream": false
}
```
**Response (`application/json`)**:
```json
{
  "response": "To schedule an emergency triage...",
  "provider": "groq",
  "model": "llama-3.3-70b-versatile"
}
```

**Mode B: Real-Time SSE Streaming (`stream: true`)**
```json
{
  "message": "What is the procedure for scheduling an emergency triage?",
  "stream": true
}
```
**Response Header**: `Content-Type: text/event-stream`
**Stream Chunks**:
```text
data: {"chunk": "To", "provider": "groq", "model": "llama-3.3-70b-versatile", "is_final": false, "error": null}

data: {"chunk": " schedule", "provider": "groq", "model": "llama-3.3-70b-versatile", "is_final": false, "error": null}

data: {"chunk": "", "provider": "groq", "model": "llama-3.3-70b-versatile", "is_final": true, "error": null}
```

---

## Running Automated Tests
Run the test suite with `pytest`:
```bash
venv\Scripts\python.exe -m pytest -v
```

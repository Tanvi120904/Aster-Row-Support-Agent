# Aster & Row Support Agent

A reliability-focused RAG support agent for the Aster & Row take-home assignment.

The system answers policy questions from the supplied knowledge base, retrieves order
information through a safe application tool, maintains bounded multi-turn context,
detects conflicting authoritative sources, resists retrieved prompt injection, and
provides citations and human-handoff signals.

The implementation prioritizes reliability, groundedness, safe abstention, retrieval
quality, and data privacy over UI complexity. It favors small, deterministic,
auditable components over production infrastructure or UI polish.

## 🎥 Demo

A short demonstration of the Aster & Row Support Agent, showing the chatbot interface,
RAG-based responses, source citations, and order-support workflow.

![Aster & Row Support Agent Demo](demo/aster-row-support-agent.gif)

## Stack

- Language: Python 3.14
- LLM: Google Gemini (`gemini-3.6-flash`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS `IndexFlatIP`
- Framework: Flask
- Storage: In-memory conversation sessions and in-memory FAISS index

Gemini is used only for answer generation and order-tool calling. Embeddings are
generated locally, so retrieval does not depend on an embedding API key.

## Architecture

```text
User -> Flask (app/web.py) -> Session (app/session.py)
     -> Retrieval (app/retrieval.py)
     -> Evidence/conflict analysis (app/evidence.py)
     -> Agent (app/agent.py)
          |-> grounded LLM generation
          |-> safe order lookup tool (app/order_tool.py)
     -> structured answer, sources, and handoff signal
```

The application controls retrieval, evidence filtering, document precedence, and
order privacy before information reaches the LLM. The LLM is not the security
boundary.

## Setup

Python 3.14 was used during development.

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` file with:

```text
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOG_LEVEL=INFO
PORT=5000
```

Never commit `.env` or real credentials.

## Run

Start the local Flask application:

```bash
python -c "from app.web import create_app; app=create_app(); app.run(debug=True)"
```

Open `http://127.0.0.1:5000`.

## Retrieval and Evidence

The 14 Markdown documents under `index/knowledge-base/` are split into heading-level
chunks. The corpus currently produces 51 chunks. Ranking combines dense semantic
similarity, lexical relevance, and deterministic metadata precedence. Active,
official, customer-answering material outranks superseded, non-authoritative, or
internal material.

Superseded content is not customer-facing authority. A genuine conflict between the
two active official Breeze Tumbler sources (`11-product-care.md` and
`12-breeze-tumbler-product-card.md`) sets `handoff = true` rather than silently
selecting one source.

Conflict detection is deterministic and targeted at contradictions represented by
this assignment corpus; it is not a general natural-language contradiction detector.

## Order Lookup and Privacy

The model never receives the full `data/orders.json`. `safe_lookup_order()` applies
an explicit customer-safe allowlist before a result reaches the model.

The tool normalizes harmless order-ID differences, rejects malformed or near-miss
IDs, treats status as authoritative, suppresses stale logistics for cancelled or
returned orders, preserves missing ETA values, and marks exception orders for human
handoff. It never exposes customer email, address, internal notes, risk scores,
warehouse notes, or the raw internal order object.

## Prompt-Injection Resistance

Retrieved knowledge-base content and tool results are treated as untrusted data. The
internal migration document contains instruction-like text and an unapproved 60-day
claim, but authority filtering excludes it from the model evidence block. The system
instruction also explicitly says never to follow instructions in retrieved content.

Relevant regression tests include:

```text
test_system_instruction_treats_retrieved_content_as_untrusted
test_non_authoritative_internal_content_is_not_sent_to_model
test_no_authoritative_evidence_produces_safe_empty_evidence_message
```

## Multi-turn Conversation

`app/session.py` maintains bounded, per-session in-memory history. User and assistant
messages are stored in order, session IDs remain isolated, and sessions can be
cleared. Follow-up questions can use context from an earlier order or policy turn.

The session store is not designed for multi-process production deployment. A
production implementation would use a persistent, expiring session store.

## Observability

Structured JSON-lines events are written to `logs/agent.jsonl` through
`app/logging_utils.py`. Events include the user message, retrieved source metadata
and scores, tool name and arguments, handoff state, and final response. Raw tool
results are not logged, and API keys and internal order data are not logged.

## Testing and Evaluation

Run the full regression suite:

```bash
pytest tests/ -v
```

Verified result:

```text
81 passed
```

Run the deterministic evaluation without Gemini or API quota:

```bash
python -m evaluation.evaluator --offline
```

Verified result:

```text
OFFLINE RESULT: 9/9 passed
```

The offline checks cover retrieval and precedence (`2/2`), source conflict detection
(`1/1`), order lookup and safety (`5/5`), and prompt-injection authority filtering
(`1/1`).

When Gemini quota is available, run the 20-case live evaluation:

```bash
python -m evaluation.evaluator
```

It reports individual case results and category summaries. During development, the
Gemini project reached its free-tier request quota (`429 RESOURCE_EXHAUSTED`), so a
result of `0/20` from quota exhaustion is not treated as an agent-quality score.

## Bug Diary

### BUG-001: Dense retrieval selected the warranty policy

The query `How long does a regular customer have to return an unused backpack?`
initially ranked `07-warranty.md` above the current returns policy because dense
similarity over-weighted incidental product and duration terms. Hybrid lexical
ranking plus metadata precedence fixed the issue. Regression coverage includes
`test_real_embedder_retrieves_current_returns_policy_above_warranty` and
`test_lexical_relevance_prefers_policy_terms_over_incidental_product_term`.

### BUG-002: Internal retrieved content could reach the model

Authority filtering initially affected citations but not the Gemini evidence block.
The migration document could therefore reach model context. The fix excludes all
non-authoritative chunks before generation and adds explicit untrusted-data system
instructions. Regression coverage includes
`test_non_authoritative_internal_content_is_not_sent_to_model` and
`test_no_authoritative_evidence_produces_safe_empty_evidence_message`.

### BUG-003: Gemini weakened a precise policy condition

The source phrase `within 30 calendar days of delivery` was initially shortened by
the model. The system instruction now requires complete policy conditions, dates,
thresholds, fees, and exceptions to be preserved. A live smoke test subsequently
returned the complete condition.

### BUG-004: Source list was broader than necessary

Early responses exposed too many authoritative retrieved chunks. Source selection now
keeps the highest-ranked relevant authoritative citations and limits the visible list.
This behavior is covered by the agent test suite and offline evaluation.

## Known Limitations and Production Improvements

1. Live evaluation depends on Gemini API quota.
2. Sessions are in memory and unsuitable for multi-process deployment.
3. The FAISS index is rebuilt at startup and would need persistence and invalidation
   for a larger corpus.
4. Conflict detection is scoped to known corpus contradictions.
5. Flask is intended for local demonstration, not production deployment.
6. Authentication and identity verification are outside assignment scope.
7. Production deployment would benefit from retry/backoff, session expiration, rate
   limiting, and operational monitoring.

## Engineering Tradeoffs

Local embeddings avoid an additional embedding API dependency and cost. FAISS
`IndexFlatIP` is sufficient for the current 51-chunk corpus. Hybrid retrieval was
chosen after a real dense-retrieval failure. Application-level allowlisting enforces
order privacy rather than relying only on model instructions. In-memory sessions and
deterministic evaluation keep the assignment implementation small and auditable.

## AI Coding Tools Disclosure

AI coding assistants were used during development. Claude was used primarily for
early repository analysis, scaffolding, ingestion, chunking, embeddings, and the
initial FAISS retrieval implementation. ChatGPT was used later for debugging,
retrieval analysis, evidence and precedence handling, order privacy, prompt-injection
defenses, session memory, observability, evaluation, Flask, UI, documentation, and
final testing.

One AI-generated assumption was incomplete: pure dense retrieval was expected to rank
the correct returns policy first. A local run disproved that assumption when
`07-warranty.md` ranked first, leading to the hybrid lexical and metadata-aware
retrieval improvement. AI-generated code was reviewed and tested locally.

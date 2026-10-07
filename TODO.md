# Roadmap & Optimization Backlog: EPFO RAG System

This document outlines the prioritized improvements for the **EPFO Circulars & Statutory Manuals RAG System**. It is divided into two distinct execution phases:
1. **Phase 1: Hugging Face Spaces Deployment** (Near-term optimizations tailored for single-container, CPU-constrained, zero-cost hosting).
2. **Phase 2: Enterprise Cloud Migration** (Long-term architectural evolution for deployment on AWS, GCP, Azure, or on-premises Kubernetes).

---

## 🚀 Phase 1: Immediate Improvements for Hugging Face Spaces

> **Target Environment**: Hugging Face Spaces (Streamlit SDK, 2 vCPU, 16 GB RAM, CPU Basic Tier).  
> **Goal**: Sub-second search, minimal RAM footprint, zero cold-start delay, $0/month infrastructure cost.

### 1. Storage & Memory Optimization
- [ ] **Precompute and bundle the compressed BM25 index in Git LFS**:
  - *Current issue*: If `vector_store/data_index/faiss_index.bm25.json.gz` is absent, the first search freezes for 45–60 seconds while pure Python tokenizes 74,317 passages on 2 shared vCPUs.
  - *Action*: Generate `faiss_index.bm25.json.gz` (~15–20 MB compressed) during build/ingestion and track it via `.gitattributes` under Git LFS.
  - *Files touched*: [`retriever.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/retriever.py), [`.gitattributes`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/.gitattributes).
- [ ] **Migrate 105 MB metadata JSON to embedded SQLite (`passages.db`)**:
  - *Current issue*: Loading `faiss_index.texts.json` via `json.load()` inflates Python heap RAM by 500 MB–800 MB on container boot.
  - *Action*: Create a lightweight SQLite database storing `(doc_id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)`. When FAISS returns candidate IDs, fetch only the matching rows via `SELECT text, metadata FROM passages WHERE id IN (...)`.
  - *Expected win*: Process startup drops to <1 second; runtime RAM drops from ~800 MB to <50 MB.
  - *Files touched*: [`vector_indexer.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/vector_indexer.py), [`retriever.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/retriever.py), [`app.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/app.py).

### 2. CPU Inference & Latency Acceleration
- [ ] **Quantize Cross-Encoder to ONNX Runtime INT8**:
  - *Current issue*: Evaluating 20 candidate pairs with PyTorch `ms-marco-MiniLM-L-6-v2` takes ~1.5s–2.5s on 2 vCPUs.
  - *Action*: Convert the model to ONNX format with INT8 dynamic quantization (`onnxruntime-cpu`).
  - *Expected win*: 60%–70% drop in re-ranking latency (~350ms total) with identical ranking precision.
  - *Files touched*: [`retriever.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/retriever.py), [`config.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/config.py), [`requirements.txt`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/requirements.txt).
- [ ] **Confidence-gated fast path for statutory / exact queries**:
  - *Current issue*: Every query goes through the full Cross-Encoder pipeline regardless of confidence.
  - *Action*: If top BM25 and dense results show high reciprocal agreement (e.g. direct circular numbers like `"WSU/2022/1"` or `"Section 17"`), return top results directly without invoking the Cross-Encoder.
  - *Files touched*: [`retriever.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/retriever.py).

### 3. Code Integrity & Ingestion Consistency
- [ ] **Align FAISS index metric in `update_indexer.py`**:
  - *Current issue*: [`update_indexer.py#L224`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/update_indexer.py#L224) creates fallback indexes using `faiss.IndexFlatL2` (Euclidean), while [`vector_indexer.py#L242`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/vector_indexer.py#L242) uses `faiss.IndexFlatIP` (Cosine similarity).
  - *Action*: Standardize on `IndexFlatIP` with normalized vectors across all ingestion scripts.
  - *Files touched*: [`update_indexer.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/update_indexer.py).
- [ ] **Pre-warm models on Space startup**:
  - *Action*: Run a dummy 1-token encode and search at the bottom of [`app.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/app.py) so the first real user experiences instantaneous response.
  - *Files touched*: [`app.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/app.py).

### 4. Operational Guardrails
- [ ] **Preserve and refine shared token rate limiting**:
  - *Action*: Ensure [`rate_limiter.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/rate_limiter.py) sliding window remains active for public web sessions, with prominent UI alerts when throttled and clean guidance for BYOK (Bring Your Own Key).
  - *Files touched*: [`app.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/app.py), [`rate_limiter.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/rate_limiter.py).

---

## 🏢 Phase 2: Long-Term Enterprise Cloud Migration

> **Target Environment**: Enterprise Cloud (AWS, GCP, Azure, or On-Premise Kubernetes).  
> **Goal**: High concurrency, horizontal scaling, strict data governance, HRMS/ERP integration, SLA guarantees.

### 1. Architecture Decoupling & API Modernization
- [ ] **Extract headless FastAPI microservice**:
  - Decouple retrieval and generation out of Streamlit into a dedicated, asynchronous FastAPI service:
    - `POST /api/v1/search` (Dense + Sparse retrieval + citations)
    - `POST /api/v1/chat/completions` (Streaming grounded synthesis)
    - `POST /api/v1/documents/analyze` (OCR & noting sheet generation)
    - `GET /api/v1/health` (Liveness / readiness probes for K8s)
  - Run with `uvicorn` and multi-process workers behind an API Gateway / Nginx reverse proxy.
- [ ] **Build enterprise frontend application**:
  - Create a production React / Next.js or Angular interface with organization branding, audit logs, and responsive layout, using Streamlit only for internal admin/data labeling.

### 2. Distributed Vector & Document Infrastructure
- [ ] **Migrate from embedded FAISS to a managed vector database**:
  - Transition vector storage to **Qdrant**, **Milvus**, or **pgvector** (PostgreSQL):
    - Multi-node clustering and automatic sharding.
    - Native hybrid search (Dense vectors + Sparse BM25/SPLADE in a single engine).
    - Dynamic payload filtering (e.g. filter strictly by year, circular category, or authority level).
- [ ] **Migrate metadata and user sessions to PostgreSQL**:
  - Store indexed passages, user search history, feedback ratings (thumbs up/down), and bookmark collections in managed Cloud SQL / RDS PostgreSQL with read replicas.

### 3. Private & Air-Gapped LLM Serving
- [ ] **Self-host open-weights models (vLLM / TGI)**:
  - Deploy `Qwen2.5-7B-Instruct` or `Llama-3.1-8B-Instruct` on private cloud GPU instances (AWS g5.xlarge or GCP L4) using **vLLM** for continuous batching and high token throughput.
  - Eliminate external Hugging Face API rate limits, third-party network hops, and vendor lock-in.
- [ ] **Zero-egress & compliance isolation**:
  - Ensure all prompts, circular queries, and uploaded organizational files remain strictly inside the corporate VPC with no external internet egress.

### 4. Enterprise Identity, Security & RBAC
- [ ] **SSO & IAM Integration**:
  - Implement Single Sign-On via OpenID Connect (OIDC) / SAML 2.0 with corporate identity providers (Microsoft Entra ID / Okta / Google Workspace).
- [ ] **Role-Based Access Control (RBAC)**:
  - Restrict access to confidential circular drafts or internal audit manuals based on user roles (e.g., Regional PF Commissioner vs. Enforcement Officer vs. General Staff).
- [ ] **Comprehensive Audit Trail**:
  - Log user queries, retrieved document IDs, generated summaries, and export activities to centralized SIEM tools (Splunk, Datadog, or CloudWatch) for compliance auditing.

### 5. Integrations & Omnichannel Access
- [ ] **Intranet & Enterprise Bot Connectors**:
  - Expose webhook and bot connectors for **Microsoft Teams**, **Slack**, and internal EPFO employee intranet portals.
- [ ] **Automated Circular Ingestion Pipeline**:
  - Set up a scheduled worker (Airflow, Temporal, or AWS Lambda / EventBridge) that runs [`fetch.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/fetch.py) and [`update_indexer.py`](file:///C:/Users/IT/Documents/GitHub/pf-circulars/update_indexer.py) weekly to ingest new circulars automatically without manual developer intervention.

### 6. Observability & Continuous RAG Evaluation
- [ ] **Continuous RAG evaluation (Ragas / TruLens)**:
  - Benchmark context precision, context recall, and groundedness against a golden test set of 100+ statutory questions on every code/index release.
- [ ] **APM & Tracing**:
  - Implement OpenTelemetry tracing across the retrieval waterfall: vector search latency, BM25 lookup time, re-ranking time, and LLM Time to First Token (TTFT).

---

## 📊 Quick Comparison Matrix

| Area | Phase 1: Hugging Face Spaces | Phase 2: Enterprise Cloud |
| :--- | :--- | :--- |
| **Architecture** | Self-contained Monolith (Streamlit) | Microservices (FastAPI + React/Next.js) |
| **Vector Engine** | In-Process `faiss-cpu` + Git LFS | Distributed Vector DB (Qdrant / pgvector) |
| **Metadata DB** | Embedded SQLite (`passages.db`) | Managed Cloud PostgreSQL (RDS / Cloud SQL) |
| **Sparse Engine** | Pre-generated BM25 Gzip Cache | Database-native Hybrid / OpenSearch / Tantivy |
| **Re-ranking** | ONNX Runtime INT8 on CPU | GPU TEI (Text Embeddings Inference) Worker |
| **LLM Inference** | Serverless HF Provider (Zero GPU cost) | Private vLLM Cluster / Private VPC Endpoint |
| **Authentication** | Public with Rate Limiting / Session BYOK | Corporate SSO (Entra ID / Okta) + RBAC |
| **Hosting Cost** | **$0 / month** (Free tier) to **~$9/mo** | **~$300 - $1,500 / month** (Scalable infra) |

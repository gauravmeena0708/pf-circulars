"""React research workspace API. Run: uvicorn web_api.app:app --port 7860."""
from datetime import date
import asyncio
from contextlib import asynccontextmanager, suppress
from functools import lru_cache
import io
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Header
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from policy_registry import registered_policies
from policy_history import PolicyHistory
from policy_router import route_retrieval

ROOT = Path(__file__).resolve().parents[1]
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
LOG = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app):
    from .workspace import FILES, LOCK, TTL
    async def cleanup():
        while True:
            await asyncio.sleep(60)
            with LOCK:
                expired = [key for key, value in FILES.items() if time.monotonic()-value['created'] > TTL]
                for key in expired:
                    del FILES[key]
    task = asyncio.create_task(cleanup())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(title='EPFO Research Workspace', version='1.0.0', lifespan=lifespan)
_lock = threading.RLock()
_packets = {}


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=6000)
    policy: str = 'auto'
    as_of: date = Field(default_factory=date.today)


class AnswerRequest(BaseModel):
    language: str = 'English'


def check_day(day):
    if not date(1952, 1, 1) <= day <= date.today():
        raise HTTPException(422, 'Choose an evidence date between 1952 and today.')


def policy_by_id(id):
    policy = next((p for p in registered_policies() if p.id == id), None)
    if policy is None:
        raise HTTPException(404, 'Policy not found.')
    return policy


@lru_cache(maxsize=1)
def resources():
    import config
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from vector_indexer import load_faiss_binary_index
    from passage_store import open_passage_store
    with _lock:
        directory = str(ROOT / config.DEFAULT_INDEX_DIR / 'data_index')
        model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.EMBEDDING_DEVICE)
        index = load_faiss_binary_index(directory, model)
        store = open_passage_store(directory)
        if index is None or store is None or index.ntotal != len(store):
            raise RuntimeError('Search index unavailable')
        reranker = None
        if config.CROSS_ENCODER_MODEL_NAME:
            try:
                reranker = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME, device=config.EMBEDDING_DEVICE)
            except Exception:
                LOG.warning('Reranker unavailable; using hybrid retrieval')
        return model, index, store, reranker, directory


def hybrid_search(question):
    import config
    from retriever import retrieve_relevant_chunks
    # Serialize CPU-heavy retrieval on a small Space; no shared question state.
    with _lock:
        model, index, store, reranker, directory = resources()
        return retrieve_relevant_chunks(question, index, store.texts, store.metadata, model,
                                        cross_encoder_model=reranker,
                                        bm25_cache_path=str(Path(directory) / 'faiss_index.bm25.json.gz'))


@lru_cache(maxsize=16)
def history_for(id):
    return PolicyHistory(policy_by_id(id).path)


def pdf_path(policy, document):
    relative = document.get('local_pdf')
    if not relative:
        return None
    path = (policy.path.parent / relative).resolve()
    if not path.is_relative_to(ROOT / 'policy_data') or not path.is_file():
        return None
    return path


@app.get('/api/catalog')
def catalog():
    path = ROOT / 'vector_store/data_index/faiss_index.passages.db'
    count = None
    if path.exists():
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            # Published indexes use contiguous zero-based passage IDs. This avoids
            # scanning the entire SQLite corpus just to paint the home screen.
            count = db.execute('select coalesce(max(id), -1) + 1 from passages').fetchone()[0]
    policies = []
    for policy in registered_policies():
        history = history_for(policy.id)
        checked = [d for d in history.documents.values() if d.get('source_file')]
        policies.append(dict(id=policy.id, label=policy.label, scope=policy.scope,
                             document_count=len(history.documents), checked_count=len(checked),
                             newest_checked=max((d['issued_on'] for d in checked), default=None)))
    return dict(policies=policies, passage_count=count, answer_enabled=bool(os.environ.get('HF_TOKEN')), today=date.today().isoformat())


@app.get('/api/policies/{id}')
def policy_snapshot(id: str, as_of: date | None = None):
    day = as_of or date.today()
    check_day(day)
    policy = policy_by_id(id)
    report = history_for(id).snapshot(day.isoformat())
    for doc in report['documents']:
        doc['pdf_url'] = f'/api/policies/{id}/documents/{doc["id"]}/pdf' if pdf_path(policy, doc) else None
    return dict(label=policy.label, scope=policy.scope, **report)


@app.get('/api/policies/{id}/documents/{document_id}/pdf')
def source_pdf(id: str, document_id: str):
    policy = policy_by_id(id)
    document = history_for(id).documents.get(document_id)
    path = pdf_path(policy, document or {})
    if path is None:
        raise HTTPException(404, 'No archived original is available for this document.')
    return FileResponse(path, media_type='application/pdf', headers={'Content-Disposition': 'inline'})


@app.post('/api/search')
def search(request: SearchRequest, http_request: Request):
    from .workspace import owner
    session = owner(http_request.headers.get("x-workspace-session"))
    check_day(request.as_of)
    if not request.question.strip():
        raise HTTPException(422, 'Enter a question.')
    try:
        result = route_retrieval(request.question.strip(), request.as_of.isoformat(),
                                 lambda: hybrid_search(request.question.strip()), mode=request.policy)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    except Exception:
        LOG.exception('Search failed')
        raise HTTPException(503, 'Search is unavailable. Please retry shortly.') from None
    sources = []
    for number, chunk in enumerate(result['chunks'], 1):
        meta = dict(chunk.get('metadata', {}))
        doc_id = chunk.get('document_id') or meta.get('policy_document_id')
        local = None
        if result['policy_context'] and doc_id:
            pid = result['policy_context']['policy_id']
            policy = policy_by_id(pid)
            doc = history_for(pid).documents.get(doc_id)
            if pdf_path(policy, doc or {}):
                local = f'/api/policies/{pid}/documents/{doc_id}/pdf'
        sources.append(dict(id=number, text=chunk['text'], metadata=meta, document_id=doc_id, pdf_url=local))
    packet_id = uuid4().hex
    with _lock:
        now = time.monotonic()
        expired = [key for key, item in _packets.items() if now - item['created'] > 900]
        for key in expired:
            del _packets[key]
        if len(_packets) >= 50:
            del _packets[next(iter(_packets))]
        _packets[packet_id] = dict(created=now, owner=session, question=request.question, chunks=result['chunks'], context=result['policy_context'])
    return dict(packet_id=packet_id, question=request.question, route=result['route'], sources=sources,
                policy_context=result['policy_context'], as_of=request.as_of.isoformat())


@lru_cache(maxsize=1)
def limiter():
    import config
    from rate_limiter import SlidingWindowRateLimiter
    return SlidingWindowRateLimiter(config.LLM_RATE_LIMIT_MAX_REQUESTS, config.LLM_RATE_LIMIT_WINDOW_SECONDS)


@app.post('/api/answers/{packet_id}')
def answer(packet_id: str, request: AnswerRequest, http_request: Request, x_hf_token: str | None = Header(default=None)):
    from .workspace import owner
    session = owner(http_request.headers.get("x-workspace-session"))
    import config
    if request.language not in ('English', 'Hindi'):
        raise HTTPException(422, 'Choose English or Hindi.')
    with _lock:
        packet = _packets.get(packet_id)
    if packet is None or packet.get('owner') != session or time.monotonic() - packet['created'] > 900:
        raise HTTPException(410, 'Evidence expired. Search again before generating an answer.')
    if not packet['chunks']:
        raise HTTPException(422, 'No source evidence was found.')
    if not (x_hf_token or config.HF_TOKEN):
        raise HTTPException(503, 'AI generation is not configured. The source evidence remains available.')
    allowed, retry = (True, 0) if x_hf_token else limiter().try_acquire()
    if not allowed:
        raise HTTPException(429, 'Generation limit reached. Please retry shortly.', headers={'Retry-After': str(int(retry) + 1)})
    from answer_generator import get_llm_answer, initialize_llm, describe_llm_error
    def events():
        try:
            model = initialize_llm(hf_token=x_hf_token) if x_hf_token else initialize_llm()
            stream = get_llm_answer(packet['question'], packet['chunks'], model, stream=True,
                                    answer_language=request.language, policy_context=packet['context'])
            if isinstance(stream, str):
                stream = [stream]
            for chunk in stream:
                text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                yield json.dumps({'text': text}, ensure_ascii=False) + '\n'
            yield json.dumps({'done': True}) + '\n'
        except Exception as error:
            yield json.dumps({'error': describe_llm_error(error)}) + '\n'
    return StreamingResponse(events(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})


@app.get('/api/health')
def health():
    return {'status': 'ok'}


from .workspace import router as workspace_router
app.include_router(workspace_router)


class BodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        maximum = 100 * 1024 * 1024
        headers = dict(scope.get('headers', []))
        try:
            length = int(headers.get(b'content-length', b'0'))
        except ValueError:
            length = 0
        if length > maximum:
            from fastapi.responses import JSONResponse
            return await JSONResponse({'detail':'A request may contain at most 100 MB of files.'}, status_code=413)(scope, receive, send)
        total = 0
        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get('body', b''))
            if total > maximum:
                raise HTTPException(413, 'A request may contain at most 100 MB of files.')
            return message
        await self.app(scope, limited_receive, send)


app.add_middleware(BodyLimit)

DIST = ROOT / 'frontend/dist'
if DIST.exists():
    app.mount('/', StaticFiles(directory=DIST, html=True), name='frontend')

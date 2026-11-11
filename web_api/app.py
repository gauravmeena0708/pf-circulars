"""React research workspace API. Run: uvicorn web_api.app:app --port 7860."""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, suppress
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from epfo import currency, policy_pdfs, usage
from epfo.citation_provenance import normalise_citation_stream
from epfo.official_links import official_links, official_url
from epfo.policy_history import PolicyHistory
from epfo.policy_registry import registered_policies
from epfo.policy_router import route_retrieval

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
                expired = [key for key, value in FILES.items() if time.monotonic() - value['created'] > TTL]
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
_pdf_lock = threading.Lock()


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=6000)
    policy: str = 'auto'
    topic: str | None = Field(default=None, max_length=80)
    as_of: date = Field(default_factory=date.today)


class FeedbackRequest(BaseModel):
    packet_id: str = Field(pattern='^[a-f0-9]{32}$')
    helpful: bool
    comment: str = Field(default='', max_length=500)


class AnswerRequest(BaseModel):
    language: str = 'English'
    detail: Literal['brief', 'detailed'] = 'brief'


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
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from epfo import config
    from epfo.passage_store import open_passage_store
    from epfo.vector_indexer import load_faiss_binary_index

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


def hybrid_search(question, extra=0):
    from epfo import config
    from epfo.retriever import retrieve_relevant_chunks

    # Serialize CPU-heavy retrieval on a small Space; no shared question state.
    with _lock:
        model, index, store, reranker, directory = resources()
        # ``extra`` ranks a few more candidates (same retrieval pass) so newer circulars can be flagged.
        return retrieve_relevant_chunks(
            question,
            index,
            store.texts,
            store.metadata,
            model,
            cross_encoder_model=reranker,
            top_n_final=getattr(config, 'TOP_N_RETRIEVAL', 5) + extra if extra else None,
            bm25_cache_path=str(Path(directory) / 'faiss_index.bm25.json.gz'),
        )


@lru_cache(maxsize=16)
def history_for(id):
    return PolicyHistory(policy_by_id(id).path)


def pdf_path(policy, document):
    """A verified local copy of the original, if one is already on disk (no network)."""
    relative = document.get('local_pdf')
    if relative:
        path = (policy.path.parent / relative).resolve()
        if path.is_relative_to(ROOT / 'policy_data') and path.is_file():
            # Never serve an unresolved Git LFS pointer as an archived PDF.
            with path.open('rb') as source:
                if source.read(5) == b'%PDF-':
                    return path
    sha256 = policy_pdfs.expected_sha256(policy.path.parent, document)
    return policy_pdfs.cached_path(document.get('id'), sha256) if sha256 else None


def pdf_available(policy, document):
    """Originals are no longer stored in Git; they are fetched from the public source URL."""
    return bool(pdf_path(policy, document) or policy_pdfs.has_public_original(policy.path.parent, document))


def corpus_date():
    path = ROOT / 'vector_store/data_index/faiss_index.passages.db'
    return currency.corpus_as_of(str(path)) if path.exists() else None


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
        issued = [d['issued_on'] for d in history.documents.values()]
        policies.append(
            dict(
                id=policy.id,
                label=policy.label,
                scope=policy.scope,
                first_issued=min(issued, default=None),
                last_issued=max(issued, default=None),
                document_count=len(history.documents),
                checked_count=len(checked),
                topics=history.policy.get('topics', []),
                newest_checked=max((d['issued_on'] for d in checked), default=None),
            )
        )
    latest = corpus_date()
    return dict(
        policies=policies,
        passage_count=count,
        corpus_as_of=latest.isoformat() if latest else None,
        answer_enabled=bool(os.environ.get('HF_TOKEN')),
        today=date.today().isoformat(),
    )


@lru_cache(maxsize=1)
def _division_catalog(mtime):
    return json.loads((ROOT / 'policy_data/division_catalog.json').read_text(encoding='utf-8'))


@app.get('/api/divisions')
def divisions():
    """Policy topics by issuing division, with status where a circular states it (built by scripts/build_division_catalog.py)."""
    path = ROOT / 'policy_data/division_catalog.json'
    if not path.is_file():
        raise HTTPException(404, 'The division catalog is not available.')
    catalog = dict(_division_catalog(path.stat().st_mtime_ns))
    known = {p.id: p.label for p in registered_policies()}
    # Only link collections that are registered here, so a stale catalog cannot point at a missing one.
    catalog['collections'] = known
    return catalog


@app.get('/api/policies/{id}')
def policy_snapshot(id: str, as_of: date | None = None, topic: str | None = None):
    day = as_of or date.today()
    check_day(day)
    policy = policy_by_id(id)
    try:
        report = history_for(id).snapshot(day.isoformat(), topics=[topic] if topic else None)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    for doc in report['documents']:
        doc['pdf_url'] = f'/api/policies/{id}/documents/{doc["id"]}/pdf' if pdf_available(policy, doc) else None
        doc.update(official_links({k: doc[k] for k in ('source_url', 'original_source_url') if k in doc}))
    return dict(label=policy.label, scope=policy.scope, **report)


@app.get('/api/policies/{id}/documents/{document_id}/pdf')
def source_pdf(id: str, document_id: str):
    policy = policy_by_id(id)
    document = history_for(id).documents.get(document_id) or {}
    path = pdf_path(policy, document)
    if path is None and policy_pdfs.has_public_original(policy.path.parent, document):
        try:
            # Serialised so two viewers of the same file do not download it twice.
            with _pdf_lock:
                path = policy_pdfs.ensure_original(policy.path.parent, document)
        except Exception as error:
            LOG.warning('Public original unavailable for %s: %s', document_id, error)
            # The reader's own browser can usually reach the official site even when this server cannot.
            return RedirectResponse(official_url(document['source_url']), status_code=302)
    if path is None:
        raise HTTPException(404, 'No original is available for this document.')
    return FileResponse(path, media_type='application/pdf', headers={'Content-Disposition': 'inline'})


@app.post('/api/search')
def search(request: SearchRequest, http_request: Request):
    from .workspace import owner

    session = owner(http_request.headers.get("x-workspace-session"))
    check_day(request.as_of)
    if not request.question.strip():
        raise HTTPException(422, 'Enter a question.')
    from epfo import config

    pool = []
    started = time.monotonic()

    def run_hybrid():
        ranked = hybrid_search(request.question.strip(), extra=currency.EXTRA_CANDIDATES)
        top = getattr(config, 'TOP_N_RETRIEVAL', 5)
        pool[:] = ranked[top:]
        return ranked[:top]

    try:
        result = route_retrieval(
            request.question.strip(), request.as_of.isoformat(), run_hybrid, mode=request.policy, topic=request.topic
        )
    except ValueError as error:
        usage.record('search_failed', outcome='invalid_request')
        raise HTTPException(422, str(error)) from None
    except Exception:
        LOG.exception('Search failed')
        usage.record('search_failed', outcome='unavailable')
        raise HTTPException(503, 'Search is unavailable. Please retry shortly.') from None
    # The old epfindia.gov.in site now lives at epfo.gov.in: show readers (and the model) current links.
    for chunk in result['chunks']:
        chunk['metadata'] = official_links(chunk.get('metadata', {}))
    for document in (result['policy_context'] or {}).get('documents', []):
        document.update(
            official_links({k: document[k] for k in ('source_url', 'original_source_url') if k in document})
        )
    sources = []
    for number, chunk in enumerate(result['chunks'], 1):
        meta = dict(chunk.get('metadata', {}))
        doc_id = chunk.get('document_id') or meta.get('policy_document_id')
        local = None
        if result['policy_context'] and doc_id:
            pid = result['policy_context']['policy_id']
            policy = policy_by_id(pid)
            doc = history_for(pid).documents.get(doc_id)
            if pdf_available(policy, doc or {}):
                local = f'/api/policies/{pid}/documents/{doc_id}/pdf'
        sources.append(dict(id=number, text=chunk['text'], metadata=meta, document_id=doc_id, pdf_url=local))
    # Curated histories carry their own supersession review; the general collection does not.
    freshness = None
    if result['route'] == 'hybrid':
        freshness = currency.build_currency(result['chunks'], pool, request.question, corpus_as_of=corpus_date())
    packet_id = uuid4().hex
    with _lock:
        now = time.monotonic()
        expired = [key for key, item in _packets.items() if now - item['created'] > 900]
        for key in expired:
            del _packets[key]
        if len(_packets) >= 50:
            del _packets[next(iter(_packets))]
        _packets[packet_id] = dict(
            created=now,
            owner=session,
            question=request.question,
            chunks=result['chunks'],
            context=result['policy_context'],
            currency=freshness,
            route=result['route'],
        )
    usage.record(
        'search',
        route=result['route'],
        policy=(result['policy_context'] or {}).get('policy_id'),
        chosen=request.policy,
        sources=len(sources),
        fallback=bool((result['policy_context'] or {}).get('fallback_reason')) or None,
        as_of_today=request.as_of == date.today(),
        seconds=round(time.monotonic() - started, 2),
        question=request.question.strip() if usage.questions_logged() else None,
    )
    return dict(
        packet_id=packet_id,
        question=request.question,
        route=result['route'],
        sources=sources,
        policy_context=result['policy_context'],
        currency=freshness,
        as_of=request.as_of.isoformat(),
    )


@lru_cache(maxsize=1)
def limiter():
    from epfo import config
    from epfo.rate_limiter import SlidingWindowRateLimiter

    return SlidingWindowRateLimiter(config.LLM_RATE_LIMIT_MAX_REQUESTS, config.LLM_RATE_LIMIT_WINDOW_SECONDS)


@app.post('/api/answers/{packet_id}')
def answer(
    packet_id: str, request: AnswerRequest, http_request: Request, x_hf_token: str | None = Header(default=None)
):
    from .workspace import owner

    session = owner(http_request.headers.get("x-workspace-session"))
    from epfo import config

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
        usage.record('answer', outcome='local_limit', route=packet.get('route'))
        raise HTTPException(
            429, 'Generation limit reached. Please retry shortly.', headers={'Retry-After': str(int(retry) + 1)}
        )
    from epfo.answer_generator import describe_llm_error, get_llm_answer, initialize_llm, llm_error_kind

    def events():
        started, written = time.monotonic(), []

        def finish(outcome):
            usage.record(
                'answer',
                outcome=outcome,
                route=packet.get('route'),
                policy=(packet['context'] or {}).get('policy_id'),
                language=request.language,
                detail=request.detail,
                own_token=bool(x_hf_token) or None,
                characters=sum(map(len, written)),
                seconds=round(time.monotonic() - started, 2),
            )

        try:
            options = {'hf_token': x_hf_token} if x_hf_token else {}
            if request.detail == 'detailed':
                options['max_new_tokens'] = max(config.LLM_MAX_NEW_TOKENS, 2048)
            model = initialize_llm(**options)
            stream = get_llm_answer(
                packet['question'],
                packet['chunks'],
                model,
                stream=True,
                answer_language=request.language,
                policy_context=packet['context'],
                currency=packet.get('currency'),
                answer_detail=request.detail,
            )
            if isinstance(stream, str):
                stream = [stream]
            pieces = (chunk.content if hasattr(chunk, 'content') else str(chunk) for chunk in stream)
            # Models sometimes write 【3】 or [1, 2]; readers and exports only recognise [3].
            for text in normalise_citation_stream(pieces):
                written.append(text)
                yield json.dumps({'text': text}, ensure_ascii=False) + '\n'
            # A provider failure arrives as a short explanatory text, not as an exception.
            finish(llm_error_kind(''.join(written)) or 'ok')
            yield json.dumps({'done': True}) + '\n'
        except Exception as error:
            finish('error')
            yield json.dumps({'error': describe_llm_error(error)}) + '\n'

    return StreamingResponse(
        events(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'}
    )


@app.post('/api/feedback')
def feedback(request: FeedbackRequest, http_request: Request):
    """Record whether an answer helped. The interface tells the reader the question is recorded."""
    from .workspace import owner

    session = owner(http_request.headers.get("x-workspace-session"))
    with _lock:
        packet = _packets.get(request.packet_id)
    if packet is None or packet.get('owner') != session:
        raise HTTPException(410, 'This answer is no longer available for feedback.')
    usage.record(
        'feedback',
        helpful=request.helpful,
        route=packet.get('route'),
        policy=(packet['context'] or {}).get('policy_id'),
        question=packet['question'],
        comment=request.comment.strip() or None,
    )
    return {'status': 'recorded'}


@app.get('/api/stats')
def stats():
    """Counts since the process started: searches by route and policy, answer outcomes, feedback."""
    return usage.snapshot()


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

            return await JSONResponse({'detail': 'A request may contain at most 100 MB of files.'}, status_code=413)(
                scope, receive, send
            )
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

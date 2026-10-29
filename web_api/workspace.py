"""Session-isolated uploads, document/data chat, office exports and PDF operations."""
from dataclasses import asdict
import io
import json
import math
from numbers import Real, Integral
import re
import threading
import time
from uuid import uuid4
from urllib.parse import quote
import zipfile

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix='/api')
FILES = {}
LOCK = threading.RLock()
WORK = threading.BoundedSemaphore(2)
MAX_FILE = 50 * 1024 * 1024
MAX_CACHE = 256 * 1024 * 1024
TTL = 1800
DOCX = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value, bool): return value
    if isinstance(value, Integral): return int(value)
    if isinstance(value, Real): return float(value) if math.isfinite(value) else None
    return value


def owner(value):
    if not value or not re.fullmatch('[a-f0-9]{32}', value):
        raise HTTPException(400, 'Reload the workspace to initialise your private upload session.')
    return value


def read_upload(upload, limit=MAX_FILE):
    try:
        data = upload.file.read(limit + 1)
    finally:
        upload.file.close()
    if not data or len(data) > limit:
        raise HTTPException(413, f'Choose a non-empty file up to {limit // (1024 * 1024)} MB.')
    return data


def filename(value):
    return re.sub(r'[\r\n\x00]', '', (value or 'document').replace('\\', '/').split('/')[-1])[:150]


def binary(data, name, media):
    return Response(data, media_type=media, headers={'Content-Disposition': "attachment; filename*=UTF-8''" + quote(name), 'Cache-Control': 'no-store'})


def get_file(id, session):
    owner(session)
    with LOCK:
        entry = FILES.get(id)
        if not entry or entry['owner'] != session or time.monotonic() - entry['created'] > TTL:
            if entry and time.monotonic() - entry['created'] > TTL:
                del FILES[id]
            raise HTTPException(404, 'Upload expired or is unavailable in this browser session.')
        return entry


def store(entry):
    with LOCK:
        now = time.monotonic()
        for id in list(FILES):
            if now - FILES[id]['created'] > TTL:
                del FILES[id]
        if sum(e['bytes'] for e in FILES.values()) + entry['bytes'] > MAX_CACHE or len(FILES) >= 24:
            raise HTTPException(503, 'The upload workspace is busy. Remove a file or retry later.')
        id = uuid4().hex
        FILES[id] = dict(entry, created=now, history=[])
        return id


def check_pdf(data, pages=250, raster=False):
    import fitz
    try:
        with fitz.open(stream=data, filetype='pdf') as pdf:
            if pdf.needs_pass:
                raise ValueError('Unlock the password-protected PDF before uploading.')
            if not 1 <= len(pdf) <= pages:
                raise ValueError(f'This operation supports PDFs with 1–{pages} pages.')
            if raster and any(page.rect.width * page.rect.height * (200 / 72) ** 2 > 20_000_000 for page in pdf):
                raise ValueError('A page is too large for OCR/rendering. Resize the page first.')
            return len(pdf)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(422, str(error)) from None


class OCRReader:
    def __init__(self, language='eng+hin'):
        self.language = language

    def readtext(self, image, **kwargs):
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.fromarray(image), lang=self.language, timeout=45).splitlines()


@router.post('/uploads')
def upload(file: UploadFile = File(...), x_workspace_session: str | None = Header(default=None)):
    session = owner(x_workspace_session)
    name = filename(file.filename)
    data = read_upload(file, 25 * 1024 * 1024 if name.lower().endswith('.csv') else MAX_FILE)
    if not WORK.acquire(blocking=False):
        raise HTTPException(503, 'Two documents are being processed. Please retry shortly.')
    try:
        if name.lower().endswith('.csv'):
            from data_assistant import load_csv_dataframe, generate_dataset_profile
            frame = load_csv_dataframe(data, name)
            size = int(frame.memory_usage(deep=True).sum())
            if size > 64 * 1024 * 1024:
                raise HTTPException(413, 'This dataset expands beyond the 64 MB analysis limit. Upload a smaller CSV.')
            profile = json_safe(asdict(generate_dataset_profile(frame)))
            id = store(dict(owner=session, name=name, kind='csv', frame=frame, profile=profile, data=data, bytes=len(data)+size))
            return dict(id=id, name=name, kind='csv', profile=profile, preview=json.loads(frame.head(100).to_json(orient='records', date_format='iso')), warnings=[])
        if not name.lower().endswith('.pdf'):
            raise HTTPException(422, 'Choose a PDF document or a CSV dataset.')
        pages = check_pdf(data, pages=100, raster=True)
        from document_assistant import extract_pdf_text
        result = extract_pdf_text(data, ocr_reader_factory=OCRReader, ocr_dpi=180)
        id = store(dict(owner=session, name=name, kind='pdf', text=result.text, data=data, bytes=len(data)+len(result.text.encode())))
        return dict(id=id, name=name, kind='pdf', page_count=pages, scanned_page_count=result.scanned_page_count,
                    text=result.text, warnings=result.warnings)
    except HTTPException:
        raise
    except Exception as error:
        from document_assistant import DocumentExtractionError
        from data_assistant import DataExtractionError
        if isinstance(error, (DocumentExtractionError, DataExtractionError, ValueError)):
            raise HTTPException(422, str(error)) from None
        raise HTTPException(422, 'The file could not be processed. Check its format and try a smaller file.') from None
    finally:
        WORK.release()


@router.delete('/uploads/{id}')
def remove_upload(id: str, x_workspace_session: str | None = Header(default=None)):
    get_file(id, x_workspace_session)
    with LOCK:
        del FILES[id]
    return {'removed': True}


@router.get('/uploads/{id}/rows')
def rows(id: str, term: str = '', column: str | None = None, x_workspace_session: str | None = Header(default=None)):
    entry = get_file(id, x_workspace_session)
    if entry['kind'] != 'csv':
        raise HTTPException(422, 'This upload is not a CSV.')
    if len(term) > 300:
        raise HTTPException(422, 'Shorten the search term.')
    from data_assistant import search_dataframe
    frame = entry['frame']
    if column and column not in frame.columns:
        raise HTTPException(422, 'Unknown column.')
    matches, count, breakdown = search_dataframe(frame, term)
    distribution = frame[column].fillna('(blank)').astype(str).value_counts().head(20).to_dict() if column else {}
    return dict(total_matches=count, column_matches=breakdown, rows=json.loads(matches.head(100).to_json(orient='records', date_format='iso')), distribution={str(k):int(v) for k,v in distribution.items()})


@router.get('/uploads/{id}/download')
def download_upload(id: str, format: str = 'text', x_workspace_session: str | None = Header(default=None)):
    entry = get_file(id, x_workspace_session)
    from docx_export import create_text_document_docx, create_chat_transcript_docx
    if format == 'original':
        return binary(entry['data'], entry['name'], 'text/csv' if entry['kind']=='csv' else 'application/pdf')
    if format == 'transcript':
        return binary(create_chat_transcript_docx(entry['name']+' — research transcript', entry['history'], {'File':entry['name']}), 'transcript.docx', DOCX)
    text = entry.get('text') or json.dumps(entry['profile'], ensure_ascii=False, indent=2, default=str)
    if format == 'docx':
        return binary(create_text_document_docx(entry['name'], text), 'extracted-document.docx', DOCX)
    if format != 'text':
        raise HTTPException(422, 'Unknown export format.')
    return binary(text.encode(), 'extracted-text.txt', 'text/plain; charset=utf-8')


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=6000)
    language: str = 'English'
    mode: str = 'answer'


@router.post('/uploads/{id}/chat')
def chat(id: str, request: ChatRequest, x_workspace_session: str | None = Header(default=None), x_hf_token: str | None = Header(default=None)):
    entry = get_file(id, x_workspace_session)
    if request.language not in ('English','Hindi') or request.mode not in ('answer','note'):
        raise HTTPException(422, 'Invalid language or drafting mode.')
    from . import app as root
    import config
    token = x_hf_token or config.HF_TOKEN
    if not token:
        raise HTTPException(503, 'AI generation needs a Hugging Face token. Add one under Session settings.')
    if not x_hf_token:
        allowed, retry = root.limiter().try_acquire()
        if not allowed:
            raise HTTPException(429, 'Generation limit reached. Retry shortly.', headers={'Retry-After':str(int(retry)+1)})
    def events():
        from answer_generator import initialize_llm, get_llm_answer, describe_llm_error
        from document_assistant import select_document_context, format_conversation_history
        history = list(entry['history'])[-8:]
        response = ''
        try:
            model = initialize_llm(hf_token=token)
            question = request.question
            if request.mode == 'note':
                question = 'Draft a formal office noting sheet with facts, issue for consideration and proposed action. Clearly mark any missing information. Task: '+question
            if entry['kind'] == 'csv':
                from data_assistant import prepare_dataframe_llm_context
                context = prepare_dataframe_llm_context(entry['frame'], question, max_context_chars=10000)
                chunks = [{'text':context, 'metadata':{'title':entry['name'], 'page_provenance':'document_only'}}]
            else:
                text, shortened = select_document_context(entry['text'], question, 12000)
                chunks = [{'text':text, 'metadata':{'title':entry['name'], 'page_provenance':'document_only', 'evidence_status':'Uploaded document; refer to the embedded physical page markers'}}]
            stream = get_llm_answer(question, chunks, model, stream=True, answer_language=request.language,
                                    conversation_context=format_conversation_history(history, max_chars=1500, max_messages=6))
            for part in ([stream] if isinstance(stream,str) else stream):
                text = part.content if hasattr(part,'content') else str(part)
                response += text
                yield json.dumps({'text':text},ensure_ascii=False)+'\n'
            with LOCK:
                entry['history'] = (history+[{'role':'user','content':request.question},{'role':'assistant','content':response}])[-20:]
            yield json.dumps({'done':True})+'\n'
        except Exception as error:
            yield json.dumps({'error':describe_llm_error(error)})+'\n'
    return StreamingResponse(events(), media_type='application/x-ndjson', headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})


class ResearchExport(BaseModel):
    packet_id: str
    answer: str = Field(default='', max_length=60000)


@router.post('/exports/research')
def research_export(request: ResearchExport, x_workspace_session: str | None = Header(default=None)):
    from . import app as root
    with root._lock:
        packet = root._packets.get(request.packet_id)
    if not packet or packet.get('owner') != owner(x_workspace_session) or time.monotonic()-packet['created']>900:
        raise HTTPException(410, 'Evidence expired. Search again before exporting.')
    from docx_export import create_research_report_docx
    sources = []
    for chunk in packet['chunks']:
        metadata = chunk['metadata']
        page = metadata.get('page_number')
        provenance = metadata.get('page_provenance', 'unverified')
        source = metadata.get('source') or metadata.get('circular_no') or 'EPFO document'
        source += f' | Page provenance: {provenance}'
        if page and provenance not in ('document_only', 'unverified'):
            source += f' | Physical page: {page}'
        sources.append(dict(metadata, source=source, text=chunk['text'], url=metadata.get('english_pdf_link')))
    qualifier = ('\n\nEvidence cutoff: '+packet['context']['as_of']+
                 '. Evidence is incomplete; applicability is not certified.') if packet['context'] else '\n\nStandard hybrid search; the history date cutoff is not applied.'
    return binary(create_research_report_docx(packet['question'], request.answer+qualifier, sources), 'epfo-research-report.docx', DOCX)


def zipped(items):
    result = io.BytesIO()
    with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in items:
            archive.writestr(name,data)
    return result.getvalue()


@router.post('/pdf-tools')
def pdf_tool(operation: str = Form(...), files: list[UploadFile] = File(...), pages: str = Form('', max_length=1000), rotation: int = Form(90), dpi: int = Form(150), language: str = Form('eng+hin')):
    import pdf_utils as tools
    if operation not in {'merge','extract','split','rotate','compress','images','render','ocr'}:
        raise HTTPException(422, 'Choose a supported PDF operation.')
    if not 1 <= len(files) <= 20 or (operation not in {'merge','images'} and len(files)!=1) or (operation=='merge' and len(files)<2):
        raise HTTPException(422, 'Choose two or more files for merge, or one file for this operation (maximum 20).')
    if dpi not in (100,150,200) or language not in ('eng','hin','eng+hin'):
        raise HTTPException(422, 'Choose a supported resolution and OCR language.')
    if not WORK.acquire(blocking=False):
        raise HTTPException(503, 'Two documents are being processed. Retry shortly.')
    try:
        inputs = [(filename(f.filename), read_upload(f)) for f in files]
        if operation != 'images':
            total_pages = sum(check_pdf(data, pages=30 if operation=='ocr' else 50 if operation=='render' else 250, raster=operation in {'ocr','render'}) for _,data in inputs)
            if total_pages>300:
                raise HTTPException(422, 'A combined operation may contain at most 300 pages.')
        data = inputs[0][1]
        if operation=='merge':
            output=tools.merge_pdfs([data for _,data in inputs])
        elif operation=='extract':
            if not pages.strip():raise HTTPException(422, 'Enter page numbers or ranges, for example 1,3-5.')
            output=tools.split_pdf(data,pages)
        elif operation=='split':return binary(zipped(tools.split_pdf_to_individual_pages(data)),'split-pages.zip','application/zip')
        elif operation=='rotate':output=tools.rotate_pdf_pages(data,rotation,pages or None)
        elif operation=='compress':output=tools.compress_pdf(data)[0]
        elif operation=='images':output=tools.images_to_pdf(inputs)
        elif operation=='render':return binary(zipped(tools.pdf_to_images(data,dpi=dpi)),'page-images.zip','application/zip')
        else:
            # Tesseract's PDF renderer preserves word positions and Unicode text.
            import fitz,pytesseract
            from PIL import Image
            text=[]
            with fitz.open(stream=data,filetype='pdf') as original, fitz.open() as out:
                for i,page in enumerate(original):
                    image=Image.open(io.BytesIO(page.get_pixmap(dpi=dpi,alpha=False).tobytes('png')))
                    searchable=pytesseract.image_to_pdf_or_hocr(image,extension='pdf',lang=language,timeout=45)
                    with fitz.open(stream=searchable,filetype='pdf') as one:
                        out.insert_pdf(one);text.append(f'--- Page {i+1} ---\n'+one[0].get_text())
                output=out.tobytes(deflate=True)
            from docx_export import create_text_document_docx
            extracted = '\n\n'.join(text)
            return binary(zipped([('searchable.pdf',output),('extracted-text.txt',extracted.encode()),
                                  ('extracted-document.docx',create_text_document_docx(inputs[0][0],extracted))]),'ocr-document.zip','application/zip')
        return binary(output,operation+'-result.pdf','application/pdf')
    except HTTPException:
        raise
    except Exception as error:
        if isinstance(error,(tools.PDFProcessingError,ValueError)):
            raise HTTPException(422,str(error)) from None
        raise HTTPException(422,'Processing failed. Check the file, page range or OCR language and try again.') from None
    finally:
        WORK.release()

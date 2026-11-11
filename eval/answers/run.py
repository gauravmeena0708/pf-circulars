"""Check generated answers against expected facts, citations and date limits.

Each case in cases.json is sent through the same API routes the interface uses
(/api/search, then /api/answers), and the answer is checked by fixed rules: the right
policy was chosen, the expected source was retrieved and cited, each expected fact is
stated, nothing listed as wrong is stated, citations refer to real sources, and the
answer is in the requested language. No model judges another model.

    python eval/answers/run.py                      # all cases; needs HF_TOKEN and uses paid inference
    python eval/answers/run.py --retrieval-only     # routing and sources only; no inference
    python eval/answers/run.py --case dr-2025-07 --case cpps-any-bank
    python eval/answers/run.py --out report.json    # keep the answers for review

Exit status is 1 if any case fails. The expectations are drafts written from the cited
source pages until a subject expert marks each case reviewed.
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks

SESSION = 'e' * 32
RATE_LIMITED = 'rate-limiting requests (429)'  # wording from answer_generator.describe_llm_error
RATE_LIMIT_ATTEMPTS = 5
RATE_LIMIT_WAIT = 30
STILL_LIMITED = 'the provider was still rate-limiting after repeated attempts'


def read_answer(response):
    """Join the streamed answer; return (text, error)."""
    parts, error = [], None
    for line in response.iter_lines():
        if not line:
            continue
        event = json.loads(line)
        if 'text' in event:
            parts.append(event['text'])
        if 'error' in event:
            error = event['error']
    return ''.join(parts), error


def run_case(client, case, token, generate=True, detail='brief'):
    result = {'id': case['id'], 'question': case['question'], 'review': case.get('review', 'draft')}
    body = {
        'question': case['question'],
        'policy': case.get('policy', 'auto'),
        'as_of': case.get('as_of') or date.today().isoformat(),
    }
    if case.get('topic'):
        body['topic'] = case['topic']
    response = client.post('/api/search', json=body)
    if response.status_code != 200:
        result['failures'] = [f'search failed: {response.status_code} {response.text[:200]}']
        return result
    search = response.json()
    result['route'] = (search.get('policy_context') or {}).get('policy_id') or 'standard search'
    result['sources'] = [
        source.get('document_id') or (source.get('metadata') or {}).get('title') for source in search['sources']
    ]
    failures = checks.check_retrieval(case, search)
    if generate and search['sources']:
        # The provider limits tokens per minute; a rate-limited attempt is repeated, not scored.
        for attempt in range(RATE_LIMIT_ATTEMPTS):
            started = time.monotonic()
            with client.stream(
                'POST',
                f"/api/answers/{search['packet_id']}",
                json={'language': case.get('language', 'English'), 'detail': detail},
                headers={'X-HF-Token': token},
            ) as stream:
                if stream.status_code != 200:
                    stream.read()
                    failures.append(f'answer request failed: {stream.status_code} {stream.text[:200]}')
                    answer, error = '', None
                else:
                    answer, error = read_answer(stream)
            if RATE_LIMITED not in (answer or '') + (error or '') or attempt == RATE_LIMIT_ATTEMPTS - 1:
                break
            time.sleep(RATE_LIMIT_WAIT)
        result['attempts'] = attempt + 1
        result['seconds'] = round(time.monotonic() - started, 1)
        if RATE_LIMITED in (answer or ''):
            answer, error = '', STILL_LIMITED
        result['answer'] = answer
        if error:
            failures.append(f'generation error: {error}')
        elif answer:
            failures += checks.check_answer(case, search, answer)
            result['cited'] = checks.cited_numbers(checks.normalise(answer))
    result['failures'] = failures
    return result


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('cases.json'))
    parser.add_argument('--case', action='append', help='Run only this case id (repeatable)')
    parser.add_argument(
        '--retrieval-only', action='store_true', help='Check routing and sources without generating answers'
    )
    parser.add_argument('--detail', choices=['brief', 'detailed'], default='brief')
    parser.add_argument('--out', type=Path, help='Write the full report, including answers, to this file')
    args = parser.parse_args()

    dataset = json.loads(args.cases.read_text(encoding='utf-8'))
    cases = [case for case in dataset['cases'] if not args.case or case['id'] in args.case]
    if not cases:
        parser.error('No case matches')

    os.chdir(ROOT)
    from epfo import config

    token = config.HF_TOKEN
    if not args.retrieval_only and not token:
        parser.error('HF_TOKEN is not set; use --retrieval-only or add it to .env')
    from fastapi.testclient import TestClient

    from web_api.app import app

    client = TestClient(app, headers={'X-Workspace-Session': SESSION})

    results, throttled = [], 0
    for number, case in enumerate(cases, 1):
        result = run_case(client, case, token, generate=not args.retrieval_only, detail=args.detail)
        results.append(result)
        status = 'PASS' if not result['failures'] else 'FAIL'
        print(
            f"[{number}/{len(cases)}] {status} {case['id']}" + ''.join(f'\n      - {f}' for f in result['failures']),
            flush=True,
        )
        # Two cases in a row that stay rate-limited mean the daily token allowance is used up.
        # That allowance is shared with the live Space, so stop instead of waiting it out.
        throttled = throttled + 1 if any(STILL_LIMITED in f for f in result['failures']) else 0
        if throttled == 2:
            print(
                'Stopping: the provider is still rate-limiting. The daily token allowance is probably '
                'exhausted; the remaining cases were not run.',
                flush=True,
            )
            break

    failed = [r['id'] for r in results if r['failures']]
    summary = {
        'dataset_status': dataset['dataset_status'],
        'mode': 'retrieval only' if args.retrieval_only else f'generated answers ({config.LLM_REPO_ID}, {args.detail})',
        'cases': len(results),
        'not_run': len(cases) - len(results),
        'passed': len(results) - len(failed),
        'failed_cases': failed,
        'reviewed_cases': sum(r['review'] == 'reviewed' for r in results),
        'not_measured': ['claims outside the listed facts', 'completeness of the answer', 'legal applicability'],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        args.out.write_text(
            json.dumps({'summary': summary, 'results': results}, indent=1, ensure_ascii=False), encoding='utf-8'
        )
        print(f'Report written to {args.out}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

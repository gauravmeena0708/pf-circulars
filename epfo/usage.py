"""Anonymous usage events: what kinds of questions arrive, where they fail, and reader feedback.

Each event is one JSON line on the `epfo.usage` logger (the Space log) and, when
USAGE_LOG_FILE is set, appended to that file. Counters since the process started are
kept for /api/stats. No session id, token or address is recorded.

Question text is recorded only (a) with feedback, where the interface tells the reader
so, or (b) for every search when USAGE_LOG_QUESTIONS=true.
"""

import json
import logging
import os
import sys
import threading
import time
from collections import Counter
from datetime import UTC, datetime

LOG = logging.getLogger('epfo.usage')
# The server does not configure logging for this name, so give it its own output: one bare
# JSON line per event on stdout, not repeated through the root logger.
if not LOG.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter('%(message)s'))
    LOG.addHandler(_handler)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
_lock = threading.Lock()
_counts = Counter()
_started = time.time()
# Fields whose values are counted per event, e.g. "search.route.hybrid".
_COUNTED = ('route', 'policy', 'outcome', 'language', 'detail', 'helpful')


def questions_logged():
    return os.environ.get('USAGE_LOG_QUESTIONS', '').strip().lower() in ('1', 'true', 'yes')


def record(event, **fields):
    """Record one event. Values must be JSON-serialisable; None values are dropped."""
    fields = {key: value for key, value in fields.items() if value is not None}
    line = json.dumps(
        {'time': datetime.now(UTC).isoformat(timespec='seconds'), 'event': event, **fields}, ensure_ascii=False
    )
    with _lock:
        _counts[event] += 1
        for key in _COUNTED:
            if key in fields:
                _counts[f'{event}.{key}.{fields[key]}'] += 1
        path = os.environ.get('USAGE_LOG_FILE')
        if path:
            try:
                with open(path, 'a', encoding='utf-8') as out:
                    out.write(line + '\n')
            except OSError:
                LOG.warning('Usage log file is not writable')
    LOG.info(line)


def snapshot():
    """Counts since the process started; contains no question text."""
    with _lock:
        counts = dict(sorted(_counts.items()))
    return {'since': datetime.fromtimestamp(_started, UTC).isoformat(timespec='seconds'), 'counts': counts}


def reset():
    with _lock:
        _counts.clear()

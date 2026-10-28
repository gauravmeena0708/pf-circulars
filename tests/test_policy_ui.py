from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_policy_controls_and_review_render_without_session_errors():
    root = str(Path(__file__).resolve().parents[1])
    app = AppTest.from_string(f'''
import sys
sys.path.insert(0, {root!r})
from policy_ui import policy_controls, render_policy_review
from policy_router import route_retrieval
mode, day = policy_controls()
result = route_retrieval("JD closed establishment", "2025-01-16", lambda: [])
render_policy_review(result["policy_context"])
''').run(timeout=20)
    assert not app.exception
    assert app.selectbox[0].value == "Automatic"
    assert app.date_input[0].label == "JD evidence as of"
    assert app.warning
    assert app.dataframe
    app.selectbox[0].select("JD member profile").run()
    assert not app.exception

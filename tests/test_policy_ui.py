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
    assert app.date_input[0].label == "Evidence as of"
    assert app.warning
    assert app.dataframe
    app.selectbox[0].select("JD member profile").run()
    assert not app.exception


def test_aadhaar_selector_supports_early_history_and_explicit_route():
    root = str(Path(__file__).resolve().parents[1])
    app = AppTest.from_string(f'''
import sys
from datetime import date
sys.path.insert(0, {root!r})
from policy_ui import policy_controls, render_policy_review
from policy_registry import mode_for_label
from policy_router import route_retrieval
mode, day = policy_controls()
result = route_retrieval("Aadhaar seeding UAN", day.isoformat(), lambda: [], mode=mode_for_label(mode))
render_policy_review(result["policy_context"])
''').run(timeout=20)
    from datetime import date
    app.selectbox[0].select("Member profile / Aadhaar")
    app.date_input[0].set_value(date(2016, 11, 21)).run()
    assert not app.exception
    assert any("Member profile / Aadhaar history" in c.value for c in app.caption)
    assert app.dataframe

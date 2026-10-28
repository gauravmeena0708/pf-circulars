from types import SimpleNamespace

from answer_generator import bounded_prompt, describe_llm_error, get_llm_answer


def failure():
    error = RuntimeError('private provider response')
    error.response = SimpleNamespace(status_code=413)
    return error


def test_413_reports_request_limit_without_permission_advice():
    message = describe_llm_error(failure())
    assert 'limit (413)' in message
    assert 'Verify' not in message
    assert 'private' not in message


def test_budget_omits_whole_pages_and_preserves_original_citation_numbers():
    chunks = [{'text': 'a' * 16000, 'metadata': {}},
              {'text': 'Checked commencement page.', 'metadata': {'evidence_status': 'curated relationship or commencement page'}}]
    prompt = bounded_prompt('rules?', chunks, limit=9000, policy_context={'relationships': [], 'limitations': []})
    assert len(prompt) <= 9000
    assert 'Source [2]' in prompt
    assert 'Source [1]' not in prompt
    assert 'omitted' in prompt
    assert len(chunks[0]['text']) == 16000


def test_nonstream_413_retries_once_with_smaller_request():
    class Model:
        def __init__(self): self.calls = []
        def invoke(self, messages):
            self.calls.append(messages[0].content)
            if len(self.calls) == 1: raise failure()
            return SimpleNamespace(content='answer [1]')
    model = Model()
    chunks = [{'text': 'evidence ' * 1100, 'metadata': {}}]
    assert get_llm_answer('rules?', chunks, model) == 'answer [1]'
    assert len(model.calls) == 2
    assert len(model.calls[1]) < len(model.calls[0])


def test_stream_413_retries_before_output_and_never_duplicates_partial_answer():
    class Model:
        def __init__(self, partial=False): self.calls = 0; self.partial = partial
        def stream(self, messages):
            self.calls += 1
            if self.calls == 1:
                if self.partial: yield 'partial'
                raise failure()
            yield 'success'
    model = Model()
    assert list(get_llm_answer('rules?', [], model, stream=True)) == ['success']
    assert model.calls == 2
    partial = Model(True)
    out = list(get_llm_answer('rules?', [], partial, stream=True))
    assert out[0] == 'partial'
    assert partial.calls == 1


def test_unicode_request_obeys_byte_budget():
    chunks = [{'text': 'अध्याय ' * 1000, 'metadata': {}}]
    prompt = bounded_prompt('नियम क्या हैं?', chunks, answer_language='Hindi', limit=9000)
    assert len(prompt.encode('utf-8')) <= 9000

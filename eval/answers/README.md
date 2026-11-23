# Answer evaluation

The cases under the other `eval/` folders check that the right source pages are retrieved.
This set checks the generated answer itself.

Each case in `cases.json` has a question, an evidence date, the policy the question should
be routed to, the document the answer must cite, the facts it must state and, where a later
circular exists, the facts it must not state because they were issued after the evidence
date. `run.py` sends each question through the same API routes the interface uses and
scores the answer by fixed rules in `checks.py`. No model judges another model.

```bash
python eval/answers/run.py --retrieval-only          # routing and sources only; free
python eval/answers/run.py --out report.json         # all cases with generated answers
python eval/answers/run.py --case dr-2025-07         # one case
```

Generating answers needs `HF_TOKEN` and uses paid inference. The provider limits tokens
per minute, so a rate-limited attempt is repeated after a wait and a full run of 31 cases
takes roughly 20 to 30 minutes. The exit status is 1 if any case fails.

## What a failure means

| Message | Meaning |
| --- | --- |
| `routed to X, expected Y` | The registry expressions in `policy_data/registry.json` sent the question elsewhere |
| `source not retrieved` | The expected document was not in the evidence handed to the model |
| `does not cite <document>` | The answer did not cite that document |
| `missing: <fact>` | None of the accepted wordings of the fact appear in the answer |
| `states: <claim>` | The answer states something listed as wrong, usually a rule issued after the evidence date |
| `cites sources that do not exist` | A citation number is higher than the number of sources |
| `citations written as 【n】` | The model used a bracket style the interface does not turn into links |
| `no numbered citation` | The answer has no `[n]` citation at all |

## Limits

- The expected facts are **drafts** written from the cited source pages. Each case stays
  `"review": "draft"` until someone who knows the subject reads the source named in the
  case, corrects the facts if needed and changes it to `"reviewed"`.
- A pass means the listed facts are present and the listed wrong claims are absent. It does
  not check other claims in the answer, completeness, or legal applicability.
- Facts are matched by regular expressions, so an unusual but correct wording can fail.
  Read the answer in the report before treating a `missing:` failure as a model error.
- Model output varies between runs; a case can pass once and fail the next time.

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

Generating answers needs `HF_TOKEN` and uses paid inference. The exit status is 1 if any
case fails.

**A full run uses a large share of the daily token allowance, which the live Space shares.**
On the Groq on-demand tier the limits for `openai/gpt-oss-20b` are 8,000 tokens a minute and
200,000 tokens a day; one case uses several thousand tokens. A rate-limited attempt is
repeated after a wait, so a full run takes 20 to 30 minutes, and two full runs in a day
exhaust the allowance and leave users of the Space without answers until it recovers. The
runner stops when two cases in a row stay rate-limited. Use `--case` for spot checks and
keep full runs for releases, or use a separate token.

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

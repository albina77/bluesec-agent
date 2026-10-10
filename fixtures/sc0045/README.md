# SC-0045 teaching fixture

`investigation_walkthrough.md` and `scenario_investigation.json` are the original
teaching materials supplied by the user from Downloads. `alert.json`,
`transcript.json`, and `expected.json` were extracted from the walkthrough.
The expected report and recorded decisions are evaluator inputs, never live-model
context. No file contains a competition access key or an LLM API key.

Run the exact recorded route without credentials or network:

```bash
uv run bluesec1-eval --fixture fixtures/sc0045 --scripted
```

This verifies nine calls and the five expected report artifacts. It is a plumbing
regression test, not evidence that an LLM can solve the case.

`--live-model` uses a paid configured LLM with the local exploratory simulator.
The model sees only the initial alert, catalog, and requested tool results. The
simulator owns the full graph privately. It never contacts the competition runtime.

The simulator uses the starter's bundled tool schemas, substring matching over
serialized graph records, deterministic timestamp/ID ordering, configurable
pagination, entity/relation lookup, and AD lookup. Time windows match any recorded
event timestamp. These behaviors are approximations, not a reverse-engineered
runtime contract. AD lookups do not expand group summaries. Full simulator results
can include labels/provenance from the export; such labels are not proof of an
attack. Neither simulated tool results nor report acceptance award official points.

The evaluation compares verdict and artifact pairs. It does not automatically
grade narrative reasoning, causal sufficiency, or benign explanations. Use real
practice results and manual trace review for those judgments.

---
name: coder-review-loop
description: Orchestrate implementation work through one persistent coder and one persistent read-only reviewer until strict acceptance or a bounded blocked result. Use only when explicitly invoked for a task the user wants implemented and independently reviewed.
---

# Coder Review Loop

Act only as the orchestrator. Do not implement, edit files, run project validation, or replace the
reviewer's judgment with your own. Give all implementation work to one coder and all acceptance
review to one read-only reviewer.

## Configure the run

Accept model and reasoning choices in natural language or as invocation settings:

```text
coder_model=gpt-5.6-terra coder_effort=medium
reviewer_model=gpt-5.6-sol reviewer_effort=high
max_cycles=5
```

Use those values as defaults when omitted. Accept other model and effort values only when the
current sub-agent tool supports the combination. Prefer the requested coder/reviewer model and reasoning settings when the collaboration interface supports them. If the current interface does not expose model, role, or reasoning-effort selection, use the available sub-agent configuration while preserving the required separation between one implementation agent and one independent read-only review agent. Report the actual available configuration used. Only return BLOCKED when the environment cannot provide separate implementation and review agents or cannot enforce the reviewer’s read-only behavior. `max_cycles` must be a positive integer and counts reviewer
verdicts of `REVISE`.

Before spawning an agent, build a self-contained task contract from the active user request. It
must include:

- every explicit requirement and acceptance criterion from the relevant user messages;
- the approved plan, when one exists;
- applicable constraints, scope boundaries, authorization limits, and requested deliverables;
- paths or contents of referenced specifications, Markdown files, and applicable `AGENTS.md`
  instructions that are already known; and
- the selected agent configurations and maximum cycle count.

Do not include unrelated historical conversation. Give each atomic requirement a stable ID such
as `R1`, `R2`, and preserve those IDs for the entire run.

If the collaboration tools needed to spawn, message, and wait for agents are unavailable, stop
with `BLOCKED`. Do not perform the work in the orchestrator as a fallback.

## Run the coder

Spawn exactly one agent named `coder` using the worker role, selected coder model, and selected
reasoning effort. Use a context-isolated spawn when model overrides require it, and include the
complete task contract in the prompt. Tell the coder that it owns implementation and validation
for this task, is not alone in the codebase, must preserve unrelated user changes, and must not
revert or overwrite work outside its ownership.

The coder must:

1. Read applicable repository instructions and inspect the relevant implementation before editing.
2. Record the initial repository status and identify pre-existing modified or untracked paths.
3. Implement every requirement in the task contract while keeping unrelated changes intact.
4. Run the appropriate available builds, tests, linters, type checks, or execution checks needed
   to demonstrate that the result works. It may omit an irrelevant or unavailable check only with
   a concrete explanation.
5. Inspect its final task-owned diff and return this report:

```text
STATUS: READY_FOR_REVIEW | BLOCKED
IMPLEMENTATION SUMMARY
REQUIREMENT COVERAGE (R1, R2, ...)
TASK-OWNED FILES (created / modified / deleted)
PRE-EXISTING CHANGES PRESERVED
VALIDATION EVIDENCE (exact command, result, exit status, material output)
KNOWN RISKS OR LIMITATIONS
```

Wait for the coder to finish. If it reports a recoverable problem, send a focused follow-up to the
same coder. If it is irrecoverably blocked, end the workflow as `BLOCKED` with its evidence.

## Run the reviewer

After the coder reports `READY_FOR_REVIEW`, spawn exactly one agent named `reviewer` using the
explorer role, selected reviewer model, and selected reasoning effort. Use a context-isolated spawn
when model overrides require it. Give it the complete task contract, the coder's full report, the
current cycle number, the initially dirty paths, and the task-owned file scope.

The reviewer is strictly read-only. It may use only non-mutating inspection such as repository
status, diffs, history, source and configuration reads, searches, and reading tests or call sites.
It must not:

- edit, create, delete, format, or regenerate files;
- run tests, builds, linters, formatters, type checkers, project code, or executables;
- commit, reset, checkout, stash, push, or otherwise change repository state;
- delegate any part of the review; or
- infer successful validation when the coder supplied no adequate evidence.

The reviewer must inspect the complete task-owned diff and enough surrounding code, call sites,
tests, and specifications to judge the change. It evaluates validation only from the coder's
recorded evidence. Missing, failed, or materially incomplete required validation is a reason to
return `REVISE` and instruct the coder to perform or repair it.

Require this review format:

```text
VERDICT: PASS | REVISE
CYCLE: <number>/<max_cycles>

REQUIREMENT SCORECARD
| ID | Atomic requirement | Score 0-100 | Status | Evidence |

FINDINGS
[P0-P3] Imperative title - path:line
Concrete impact and required correction.

QUALITY SCORECARD
| Metric | Score 0-100 | Evidence |
Correctness; Robustness; Maintainability; Integration consistency;
Scope discipline; Validation confidence

OVERALL SCORE: <0.70 * mean requirement score + 0.30 * mean quality score>
FILE INVENTORY: created / modified / deleted
VALIDATION EVIDENCE ASSESSMENT
REVISION INSTRUCTIONS (required for REVISE; absent for PASS)
RESIDUAL RISKS
```

Use `Met` only for a requirement score of 100, `Partial` for 1-99, and `Missing` for 0. Findings
must be concrete, actionable, introduced or left unresolved by the task changes, and cite the
smallest useful changed range. Continue reviewing the entire diff after finding the first issue.

The reviewer may issue `PASS` only when all of these are true:

- every explicit requirement scores 100;
- each quality metric scores at least 80;
- no actionable finding of any priority remains;
- the coder's evidence shows that all required compilation, execution, and tests succeeded, or
  clearly establishes a legitimate project limitation;
- the implementation is integrated, stays within scope, and preserves unrelated work; and
- the file inventory accounts for all task-owned changes.

Otherwise it must issue `REVISE`. A high overall score cannot compensate for an incomplete
requirement or an actionable finding.

## Iterate and finish

For `REVISE`, forward the reviewer's complete report and the unchanged task contract to the same
coder with a focused request to fix every finding, re-run appropriate validation, and return a new
coder report. Then send the revised coder report and updated diff scope to the same reviewer. Do
not spawn replacement agents during the loop.

Increment the cycle count for every `REVISE`. If the count reaches `max_cycles`, stop without
another coding round and return `BLOCKED` with the last scorecard and all unresolved findings. A
malformed reviewer response is not a review cycle: ask the same reviewer once to return the
required format. Treat continued protocol failure as an irrecoverable blocker.

After `PASS`, give the user a self-contained final report containing:

- `PASS`, cycles used, and the coder/reviewer model and effort settings;
- a concise implementation summary;
- the final per-requirement scores and evidence;
- quality metrics and overall score;
- coder-run validation commands and outcomes, clearly attributed to the coder;
- every task-owned created, modified, and deleted file;
- material limitations or residual risks; and
- the agent names `coder` and `reviewer`, noting that their activity is visible in the current
  task's sub-agent activity, plus this skill's global installation path.

For `BLOCKED`, use the same structure but lead with the blocking cause, unresolved requirements,
last reviewer findings, and the safest next action. Never describe a blocked result as finished or
accepted.

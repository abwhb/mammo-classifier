# CLAUDE.md

Claude Code-specific guidance for this repository. For the cross-tool project overview, read `AGENT.md` first. For architecture and rationale, read `DESIGN.md`.

---

## Project context

This is the **FAHM Biotechnology technical assessment** — a full-stack mammogram classifier (Malignant vs Benign/Normal). The hard deadline is **2026-06-07 23:59 AST**. Effective time is roughly one day of focused work.

Treat every decision through the lens of: *"would a senior reviewer call this honest and well-scoped, or thin and over-claimed?"*

## How to work in this repo

1. **Read `AGENT.md` and `DESIGN.md` before non-trivial changes** — the stack and privacy rules are locked there; don't relitigate them mid-edit.
2. **Default to the smallest credible slice.** No speculative abstractions, no auth scaffolding, no extra runtimes. Three similar lines beat a premature helper.
3. **Don't add comments that narrate what code does.** Only comment when the *why* would surprise a reader (e.g., "EfficientNet expects [3,224,224] even for grayscale — we tile the channel").
4. **No backwards-compat shims.** This is a greenfield repo with one author. Delete, don't deprecate.
5. **Never claim a feature works without running it.** For UI changes, open the page in a browser (or use the Playwright MCP). For inference, run a real DICOM through `/predict`. Tests verify code; only running verifies behavior.

## Skills you should reach for

- **`vercel:nextjs`** — for any App Router / Server Component / Server Action work in `apps/web`
- **`vercel:shadcn`** — for adding UI primitives (upload dropzone, result card, badge)
- **`vercel:vercel-functions`** — if we add anything beyond a thin proxy in `apps/web/app/api/`
- **`vercel:deployments-cicd`** — when wiring up the Vercel deploy
- **`firebase`** / **`firebase-basics`** — *only* if we pivot off GCP Cloud Run to Firebase Hosting + Cloud Functions; default plan does not use them
- **`superpowers:brainstorming`** — when the user opens a new design question (model choice, threshold strategy, etc.). Skip for execution of already-agreed plan.
- **`superpowers:writing-plans`** — when converting a design discussion into the execution plan for the next session
- **`superpowers:verification-before-completion`** — before claiming a deploy works or metrics are final

## Things to ASK before doing

- Anything that incurs cost (GCP project creation, GPU training runs, paid Vercel features)
- Anything that publishes (deploying to prod, pushing to a public GitHub repo, opening a PR)
- Choosing a different dataset, model architecture, or deploy target than what's in `DESIGN.md`
- Adding a new top-level dependency or runtime (Bun, Deno, a new ML framework)
- Anything that would persist an uploaded image beyond the request lifecycle

## Things you can do without asking

- Edit code inside `apps/api/app/`, `apps/web/`, `ml/`
- Run tests, linters, formatters
- Read DICOM samples and local model artifacts
- Run the dev servers (`uvicorn`, `pnpm dev`)
- Create/update docs (`README.md`, `REPORT.md`)

## Communication style

- Terse. Match the deadline pressure. One-line status > paragraph status.
- File:line references when pointing at code (e.g., `apps/api/app/inference.py:42`).
- When you finish a step, state what changed and what's next in one sentence. No trailing summaries.
- If something is uncertain (model accuracy, deploy quota, dataset licensing), say so explicitly — don't paper over it.

## Tooling notes specific to this repo

- **Python:** use `uv` (not `pip`) for env management — faster, lockfile-based.
- **Node:** `pnpm` only.
- **Containers:** API ships as a single Docker image; model weights fetched from GCS at start so the image stays small enough for Cloud Run cold-starts.
- **No Vercel for the API.** Inference needs longer CPU than a Vercel Function comfortably supports for batch DICOM preprocessing. Backend → Cloud Run.

## Memory

Three memories already exist under `~/.claude/projects/-Users-abdulwahabshafiq-fahm-test-task/memory/`:
- `user_role.md` — candidate context
- `project_fahm_assessment.md` — deadline & evaluation framing
- `project_scope.md` — functional requirements verbatim

Update them when scope changes; don't duplicate them here.

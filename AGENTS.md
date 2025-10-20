# Repository Guidelines

## Project Structure & Module Organization
Python neuron logic lives in `snn/` with `neuron.py` exporting `ThreeCompartmentNeuron` and parameter dataclasses. Tests sit in `tests/` and follow the `test_*.py` pattern. The real-time UI resides in `ui-vue/` (Vue 3 + Vite + TypeScript) and consumes SSE from your training backend. `docker-compose.yml` and `nats.conf` bootstrap the local NATS JetStream broker.

## Build, Test, and Development Commands
- `python -m unittest discover -s tests` runs the Python suite; use `PYTHONPATH=.` when executing from the repo root.
- `cd ui-vue && npm install` pulls UI dependencies, `npm run dev` starts the Vite dev server on http://127.0.0.1:5173, and `npm run build` outputs the production bundle to `ui-vue/dist`.
- `docker compose up -d` brings up NATS + monitoring; `docker compose down` stops the stack and preserves JetStream state under `./.data/nats`.

## Coding Style & Naming Conventions
Python modules follow PEP 8 with four-space indentation, type hints, and focused docstrings; prefer immutable config via `dataclass(frozen=True)` as in `ThreeCompartmentParams`. Keep public APIs snake_case and reuse the `ThreeCompartment*` prefix for related classes. In the UI, use PascalCase Vue components (`src/components/`), camelCase stores/actions, and the composition API with explicit return types. If you introduce linting, align it with the current Vite/Vue defaults.

## Testing Guidelines
Add regression tests to `tests/` with `unittest.TestCase`; mirror the existing `test_*` naming and assert membrane values as well as spike flags. For UI changes, adopt Vitest when available and record any manual SSE checks in the PR notes. Seed stochastic logic so suites stay deterministic.

## Commit & Pull Request Guidelines
History uses concise imperative subjects (e.g., “Add three-compartment neuron core and tests”); keep messages under ~72 characters and expand in the body if needed. In PRs, outline motivation, summarize changes, link issues, and note local results (`python -m unittest …`, `npm run build`, SSE smoke checks). Attach screenshots or GIFs for UI updates.

## Messaging & Configuration Tips
The UI targets SSE at `/events` and REST endpoints under `/api/*`; adjust `ui-vue/src/ws.ts` if hosts or ports move. Keep JetStream stream IDs stable in development to preserve replays, and rely on the bundled NATS UI at http://127.0.0.1:31311 for inspecting subjects and consumer health.

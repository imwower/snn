# Repository Guidelines

## Project Structure & Module Organization
- Core neuron logic lives in `snn/`; `neuron.py` exports `ThreeCompartmentNeuron` plus frozen parameter dataclasses.
- Python regression tests sit under `tests/` and follow the `test_*.py` pattern.
- Front-end assets are housed in `ui-vue/` (Vue 3 + Vite + TypeScript) and consume backend SSE from `/events`.
- Infrastructure files such as `docker-compose.yml` and `nats.conf` bootstrap the local NATS JetStream broker and monitoring UI.
- The `.data/` tree (placeholder datasets, JetStream state, etc.) must stay in the repo; do not delete this directory when committing changes, even if its contents are regenerated locally.

## Build, Test, and Development Commands
- `PYTHONPATH=. python -m unittest discover -s tests` runs the neuron suite headless.
- `cd ui-vue && npm install` installs UI dependencies; follow with `npm run dev` for the Vite server at http://127.0.0.1:5173.
- `npm run build` (inside `ui-vue/`) outputs the production bundle to `ui-vue/dist`.
- `docker compose up -d` launches NATS and monitoring; `docker compose down` stops services while preserving state in `./.data/nats`.

## Coding Style & Naming Conventions
- Python follows PEP 8 with four-space indentation, type hints, and immutable configs via `@dataclass(frozen=True)`.
- Exposed APIs share the `ThreeCompartment*` prefix; snake_case for functions and methods.
- Vue components use PascalCase filenames in `ui-vue/src/components/`; Composition API scripts return camelCase bindings with explicit types.
- Stay consistent with the default Vite/Vue ESLint and Prettier settings if formatting tools are introduced.

## Testing Guidelines
- Python tests use `unittest.TestCase` under `tests/`; name files and classes with `test_` prefixes.
- Assert both membrane potentials and spike flags to catch regressions; seed any stochastic logic to remain deterministic.
- Run the full suite with `PYTHONPATH=. python -m unittest discover -s tests` before pushing.

## Commit & Pull Request Guidelines
- Write concise, imperative commit subjects under ~72 characters (e.g., “Add three-compartment neuron core and tests”).
- Pull requests should state motivation, summarize changes, link issues, and include local results (`python -m unittest …`, `npm run build`).
- Attach screenshots or GIFs for UI-facing updates and note any manual SSE smoke checks.

## Messaging & Configuration Tips
- UI clients expect SSE at `/events` and REST under `/api/*`; update `ui-vue/src/ws.ts` if hosts or ports change.
- Keep JetStream stream identifiers stable to preserve replays; inspect subjects via the bundled NATS UI at http://127.0.0.1:31311.

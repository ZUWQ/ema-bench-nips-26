# Agent Guide

## Goal

Build a pure static React website for introducing the EMA-Bench paper. The site should feel like an academic project page: clear, polished, fast to scan, and focused on the paper's motivation, benchmark design, results, and resources.

## Tech Stack

- Frontend: React + TypeScript
- Build tool: Vite
- Styling: plain CSS unless the project already adds another styling system
- Static output only: no backend, no database, no runtime server dependencies

## Project Paths

- Frontend app: `ema-bench-frontend/`
- Main app entry: `ema-bench-frontend/src/App.tsx`
- Global styles: `ema-bench-frontend/src/index.css`
- App styles: `ema-bench-frontend/src/App.css`
- Static and visual assets: `ema-bench-frontend/src/assets/` or `ema-bench-frontend/public/`
- Paper introduction source: `docs/introduction.md`

## Design Direction

- Make the first screen clearly identify **EMA-Bench** and the paper topic.
- Prefer an academic project-page layout: hero, abstract, key contributions, benchmark overview, evaluation, findings, citation, and links.
- Keep typography readable and professional.
- Use real project visuals when available; otherwise use clean generated or diagrammatic assets.
- Avoid marketing-heavy copy, oversized decorative cards, and generic landing-page filler.
- Keep sections full-width and structured; use cards only for repeated items such as contributions, metrics, or results.

## Implementation Rules

- Keep components small and readable.
- Store repeated content in typed arrays or simple constants when useful.
- Do not add routing unless multiple pages are explicitly needed.
- Do not introduce new dependencies without a clear benefit.
- Ensure the site works after `npm run build`.
- Preserve existing user changes; do not reset or overwrite unrelated files.

## Commands

Run from `ema-bench-frontend/`:

```bash
npm run dev
npm run build
npm run lint
```

## Done Criteria

- The page presents the paper clearly as a static website.
- Content is aligned with `docs/introduction.md`.
- Layout is responsive on desktop and mobile.
- Build and lint pass, or any remaining issue is reported clearly.

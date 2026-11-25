# Frontend — Mustang Workshop Assistant

React 19 + TypeScript + Vite + Tailwind v4 + lucide-react.

```bash
npm install
npm run dev      # http://localhost:5173 — proxies /api to the backend on :8000
npm test         # Vitest + Testing Library
npm run build    # type-check + production build
```

Structure: `src/hooks/useChat.ts` (conversation state, session restore), `src/services/api.ts`
(typed API client), `src/components/*` (vehicle panel, message modes, safe Markdown renderer,
diagram gallery with zoom, sources, spec tables, composer).

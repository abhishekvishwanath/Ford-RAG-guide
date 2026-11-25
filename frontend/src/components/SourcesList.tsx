import { BookOpen } from 'lucide-react'
import type { Citation } from '../types/api'

export function SourcesList({
  citations,
  highlighted,
  documentTitle,
}: {
  citations: Citation[]
  highlighted: string | null
  documentTitle?: string
}) {
  if (!citations.length) return null
  return (
    <section aria-label="Manual sources" className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-2.5">
      <h5 className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
        <BookOpen className="h-3.5 w-3.5" aria-hidden /> Sources — {documentTitle ?? citations[0].document_title}
      </h5>
      <ol className="space-y-1 text-sm">
        {citations.map((c) => (
          <li
            key={c.source_id}
            id={`src-${c.chunk_id}`}
            className={`flex gap-2 rounded px-1.5 py-1 transition ${highlighted === c.source_id ? 'bg-sky-100 ring-1 ring-sky-400' : ''}`}
          >
            <span className="mt-0.5 h-fit rounded bg-sky-100 px-1.5 text-xs font-semibold text-sky-800">{c.source_id}</span>
            <span className="text-slate-700">
              <span className="font-medium text-slate-900">{c.procedure_title ?? 'Procedure'}</span>
              {c.subsection ? ` — ${c.subsection}` : ''}
              <span className="block text-xs text-slate-500">
                {c.section ? `Section ${c.section} · ` : ''}PDF p. {c.pdf_pages}
                {c.printed_pages ? ` · printed ${c.printed_pages}` : ''} · ref {c.procedure_id}
              </span>
            </span>
          </li>
        ))}
      </ol>
    </section>
  )
}

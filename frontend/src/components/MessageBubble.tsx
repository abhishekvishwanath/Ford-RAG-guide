import { useState } from 'react'
import { Ban, Check, CircleAlert, CircleHelp, Copy, RotateCcw, ShieldAlert, BookOpen } from 'lucide-react'
import type { ChatMessage } from '../types/api'
import { Markdown } from './Markdown'
import { DiagramGallery } from './DiagramGallery'
import { SourcesList } from './SourcesList'
import { SpecTables } from './SpecTables'

const MODE_STYLE = {
  asking_context: { label: 'Clarifying question', icon: CircleHelp, ring: 'border-sky-300 bg-sky-50', text: 'text-sky-800' },
  awaiting_confirmation: { label: 'Please confirm', icon: CircleAlert, ring: 'border-amber-300 bg-amber-50', text: 'text-amber-800' },
  safety_escalation: { label: 'Safety first', icon: ShieldAlert, ring: 'border-red-400 bg-red-50', text: 'text-red-800' },
  out_of_scope: { label: 'Outside the manual', icon: Ban, ring: 'border-slate-300 bg-slate-100', text: 'text-slate-700' },
  answer: { label: 'From the workshop manual', icon: BookOpen, ring: 'border-slate-200 bg-white', text: 'text-emerald-800' },
} as const

export function MessageBubble({
  message,
  isLast,
  onOption,
  onRetry,
}: {
  message: ChatMessage
  isLast: boolean
  onOption: (text: string) => void
  onRetry: () => void
}) {
  const [cited, setCited] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-slate-800 px-4 py-2.5 text-white">
          {message.content}
        </div>
      </div>
    )
  }

  if (message.error) {
    return (
      <div role="alert" className="flex items-center gap-3 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-red-900">
        <CircleAlert className="h-5 w-5 shrink-0" aria-hidden />
        <span className="flex-1">{message.content}</span>
        {isLast && (
          <button onClick={onRetry} className="inline-flex items-center gap-1 rounded-md bg-red-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-800">
            <RotateCcw className="h-4 w-4" aria-hidden /> Retry
          </button>
        )}
      </div>
    )
  }

  const r = message.response
  const mode = r?.interaction_mode ?? 'answer'
  const style = MODE_STYLE[mode]
  const Icon = style.icon
  const pq = r?.pending_question
  const isAnswer = mode === 'answer' && !!r
  const fallback = r?.validation?.warnings.some((w) => w.includes('verbatim excerpt'))

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable */
    }
  }

  return (
    <article className={`rounded-xl border px-4 py-3 shadow-sm ${style.ring}`} aria-label={style.label}>
      <header className={`mb-1.5 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide ${style.text}`}>
        <Icon className="h-4 w-4" aria-hidden /> {r ? style.label : 'Assistant'}
        {isAnswer && (
          <button
            onClick={copy}
            className="ml-auto inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium normal-case text-slate-500 hover:bg-slate-100"
            aria-label="Copy answer"
          >
            {copied ? <Check className="h-3.5 w-3.5" aria-hidden /> : <Copy className="h-3.5 w-3.5" aria-hidden />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        )}
      </header>
      <div className="text-[0.95rem] text-slate-800">
        <Markdown text={message.content} onCite={(id) => setCited(id)} />
      </div>

      {isLast && pq && pq.options.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {pq.options.map((o) => (
            <button
              key={o}
              onClick={() => onOption(o.replace(/\s*\(.*\)$/, ''))}
              className="rounded-full border border-sky-400 bg-white px-3 py-1.5 text-sm font-medium text-sky-800 hover:bg-sky-100"
            >
              {o}
            </button>
          ))}
        </div>
      )}

      {isAnswer && fallback && (
        <p className="mt-2 rounded bg-amber-50 px-2 py-1 text-xs text-amber-800">
          Showing verbatim manual excerpts because a verified summary was not available.
        </p>
      )}
      {isAnswer && r.images.length > 0 && <DiagramGallery images={r.images} />}
      {isAnswer && <SpecTables tables={r.tables} />}
      {isAnswer && <SourcesList citations={r.citations} highlighted={cited} />}
    </article>
  )
}

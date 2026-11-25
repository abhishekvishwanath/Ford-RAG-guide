import { useState, type KeyboardEvent } from 'react'
import { LoaderCircle, Send } from 'lucide-react'

export function Composer({ onSend, loading }: { onSend: (text: string) => void; loading: boolean }) {
  const [text, setText] = useState('')
  const submit = () => {
    if (!text.trim() || loading) return
    onSend(text)
    setText('')
  }
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }
  return (
    <form
      className="flex items-end gap-2 rounded-xl border border-slate-300 bg-white p-2 shadow-sm focus-within:ring-2 focus-within:ring-sky-500"
      onSubmit={(e) => {
        e.preventDefault()
        submit()
      }}
    >
      <label htmlFor="composer" className="sr-only">
        Describe the vehicle, symptom or task
      </label>
      <textarea
        id="composer"
        rows={2}
        maxLength={2000}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        placeholder="Describe the vehicle and the job, e.g. “2014 Mustang 5.0L — replace the water pump”"
        className="max-h-40 min-h-[2.75rem] flex-1 resize-y bg-transparent px-2 py-1.5 text-[0.95rem] text-slate-900 placeholder:text-slate-400 focus:outline-none"
      />
      <button
        type="submit"
        disabled={loading || !text.trim()}
        className="inline-flex h-10 items-center gap-1.5 rounded-lg bg-slate-800 px-4 font-medium text-white hover:bg-slate-900 disabled:cursor-not-allowed disabled:opacity-50"
        aria-label="Send"
      >
        {loading ? <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden /> : <Send className="h-4 w-4" aria-hidden />}
        <span className="hidden sm:inline">Send</span>
      </button>
    </form>
  )
}

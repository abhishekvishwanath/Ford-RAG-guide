import { useState } from 'react'
import { ChevronDown, Table } from 'lucide-react'
import type { TableRef } from '../types/api'

/** Tables are collapsed by default: the answer text already quotes what is relevant. */
export function SpecTables({ tables }: { tables: TableRef[] }) {
  const [open, setOpen] = useState(false)
  if (!tables.length) return null
  return (
    <section className="mt-3 rounded-lg border border-slate-200">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center gap-1.5 px-2.5 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500 hover:bg-slate-50"
      >
        <Table className="h-3.5 w-3.5" aria-hidden /> Manual tables ({tables.length})
        <ChevronDown className={`ml-auto h-4 w-4 transition ${open ? 'rotate-180' : ''}`} aria-hidden />
      </button>
      {open && (
        <div className="space-y-3 border-t border-slate-200 p-2.5">
          {tables.map((t) => (
            <div key={t.table_id} className="overflow-x-auto">
              <p className="mb-1 text-xs text-slate-500">
                <span className="mr-1 rounded bg-sky-100 px-1 font-semibold text-sky-800">{t.source_id}</span>PDF p. {t.page_number}
              </p>
              <table className="min-w-full border-collapse text-sm">
                {t.header.length > 0 && (
                  <thead>
                    <tr className="bg-slate-100">
                      {t.header.map((h, i) => (
                        <th key={i} className="px-2 py-1 text-left font-semibold">{h}</th>
                      ))}
                    </tr>
                  </thead>
                )}
                <tbody>
                  {t.rows.map((r, i) => (
                    <tr key={i} className="border-t border-slate-200">
                      {r.map((c, j) => (
                        <td key={j} className="px-2 py-1 align-top tabular-nums">{c}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

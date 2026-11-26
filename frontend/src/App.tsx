import { useEffect, useRef, useState } from 'react'
import { CircleAlert, LoaderCircle, Plus, Wrench } from 'lucide-react'
import { useChat } from './hooks/useChat'
import { api } from './services/api'
import type { Health, VehicleSelection } from './types/api'
import { MessageBubble } from './components/MessageBubble'
import { VehiclePanel } from './components/VehiclePanel'
import { Composer } from './components/Composer'

// one starter per retrieval path: full procedure, spec table, diagram, diagnostic (DTC), variant choice
const STARTERS = [
  { kind: 'Procedure + diagrams', text: '2014 Mustang 5.0L — remove and install the coolant pump' },
  { kind: 'Specification table', text: '2014 Mustang 5.0L — spark plug gap and torque specification' },
  { kind: 'Diagram lookup', text: '2014 Shelby GT500 5.8L — accessory drive belt routing diagram' },
  { kind: 'Diagnostic code', text: '2014 Mustang 3.7L — DTC P0128 is set, how do I diagnose it?' },
  { kind: 'Variant-specific', text: '2014 Mustang 5.0L — replace the front brake pads, 4 piston calipers' },
]

export default function App() {
  const { state, send, retry, newSession, updateVehicle } = useChat()
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState(false)
  const [pendingVehicle, setPendingVehicle] = useState<VehicleSelection | undefined>()
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealthError(true))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' })
  }, [state.messages.length, state.loading])

  const sendMessage = (text: string) => {
    void send(text, pendingVehicle)
    setPendingVehicle(undefined)
  }

  const saveVehicle = async (v: VehicleSelection) => {
    const deferred = await updateVehicle(v)
    if (deferred) setPendingVehicle(deferred)
  }

  const lastAssistantIdx = state.messages.map((m) => m.role).lastIndexOf('assistant')

  return (
    <div className="flex min-h-screen flex-col bg-slate-100 text-slate-900">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-800 text-white">
            <Wrench className="h-5 w-5" aria-hidden />
          </div>
          <div>
            <h1 className="text-lg font-semibold leading-tight">Mustang Workshop Assistant</h1>
            <p className="text-xs text-slate-500">
              Answers only from the {health?.model_years?.join('/') || ''} workshop manual · for qualified technicians
            </p>
          </div>
          <button
            onClick={() => {
              void newSession()
              setPendingVehicle(undefined)
            }}
            className="ml-auto inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium hover:bg-slate-50"
          >
            <Plus className="h-4 w-4" aria-hidden /> New conversation
          </button>
        </div>
        {(healthError || (health && (!health.index_ready || !health.llm_configured))) && (
          <div role="status" className="border-t border-amber-200 bg-amber-50">
            <p className="mx-auto flex max-w-7xl items-center gap-2 px-4 py-1.5 text-sm text-amber-900">
              <CircleAlert className="h-4 w-4" aria-hidden />
              {healthError
                ? 'The assistant server is not reachable.'
                : !health?.index_ready
                  ? 'The manual index is not built yet — answers are unavailable until ingestion completes.'
                  : 'No language model configured — answers will show verbatim manual excerpts.'}
            </p>
          </div>
        )}
      </header>

      <div className="mx-auto grid w-full max-w-7xl flex-1 grid-cols-1 gap-4 px-4 py-4 lg:grid-cols-[300px_1fr]">
        <div className="order-2 lg:order-1">
          <div className="lg:sticky lg:top-24">
            <VehiclePanel
              vehicle={state.vehicle}
              repair={state.repair}
              readiness={state.readiness}
              activeProcedure={state.activeProcedure}
              modelYears={health?.model_years ?? []}
              engines={health?.engines ?? []}
              onSave={saveVehicle}
            />
            {pendingVehicle && (
              <p className="mt-2 text-xs text-slate-600">Vehicle selection will be applied with your next message.</p>
            )}
          </div>
        </div>

        <main className="order-1 flex min-h-[60vh] flex-col lg:order-2">
          <div className="flex-1 space-y-4 pb-4" aria-live="polite">
            {state.messages.length === 0 && (
              <section className="rounded-xl border border-dashed border-slate-300 bg-white p-6">
                <h2 className="text-base font-semibold">What are you working on?</h2>
                <p className="mt-1 text-sm text-slate-600">
                  Describe the vehicle and the job. I'll ask for any details the manual needs to pick the right
                  procedure (model year, engine, transmission…) before searching, then show the exact steps, warnings,
                  specifications and original diagrams with page references.
                </p>
                <div className="mt-4 grid gap-2 sm:grid-cols-2">
                  {STARTERS.map((s) => (
                    <button
                      key={s.text}
                      onClick={() => sendMessage(s.text)}
                      className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-left text-sm hover:border-sky-400 hover:bg-sky-50"
                    >
                      <span className="mb-0.5 block text-[11px] font-semibold uppercase tracking-wide text-sky-700">
                        {s.kind}
                      </span>
                      {s.text}
                    </button>
                  ))}
                </div>
              </section>
            )}
            {state.messages.map((m, i) => (
              <MessageBubble key={m.id} message={m} isLast={i === lastAssistantIdx && !state.loading} onOption={sendMessage} onRetry={retry} />
            ))}
            {state.loading && (
              <div className="flex items-center gap-2 text-sm text-slate-600" role="status">
                <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden />
                {state.readiness?.status === 'READY' || state.activeProcedure
                  ? 'Searching the workshop manual…'
                  : 'Checking vehicle and task details…'}
              </div>
            )}
            <div ref={bottomRef} />
          </div>
          <div className="sticky bottom-0 bg-slate-100 pb-3 pt-1">
            <Composer onSend={sendMessage} loading={state.loading} />
            <p className="mt-1.5 text-center text-[11px] text-slate-500">
              Information retrieval from the manual — not an inspection or diagnosis. Always follow the manual's warnings.
            </p>
          </div>
        </main>
      </div>
    </div>
  )
}

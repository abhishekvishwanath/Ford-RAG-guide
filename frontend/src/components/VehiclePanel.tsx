import { useState } from 'react'
import { Car, Pencil, Wrench, X } from 'lucide-react'
import type { Readiness, SlotView, VehicleSelection } from '../types/api'

const STATUS_BADGE: Record<string, { label: string; cls: string }> = {
  confirmed: { label: 'confirmed', cls: 'bg-emerald-100 text-emerald-800' },
  user_provided: { label: 'provided', cls: 'bg-sky-100 text-sky-800' },
  inferred: { label: 'needs confirmation', cls: 'bg-amber-100 text-amber-800' },
  user_unsure: { label: 'unknown', cls: 'bg-slate-200 text-slate-700' },
}

const VEHICLE_FIELDS: [string, string][] = [
  ['model_year', 'Model year'],
  ['engine', 'Engine'],
  ['transmission', 'Transmission'],
  ['body_style', 'Body style'],
  ['variant', 'Variant'],
]
const REPAIR_FIELDS: [string, string][] = [
  ['component', 'Component'],
  ['task_type', 'Task'],
  ['dtcs', 'DTCs'],
  ['symptoms', 'Symptoms'],
]

function Row({ label, slot, required }: { label: string; slot?: SlotView; required: boolean }) {
  const badge = slot ? STATUS_BADGE[slot.status] : undefined
  const value = slot?.value == null ? null : Array.isArray(slot.value) ? slot.value.join(', ') : String(slot.value)
  return (
    <div className="flex items-start justify-between gap-2 py-1.5">
      <dt className="text-sm text-slate-500">
        {label}
        {required && !value && <span className="ml-1 text-red-600" title="Required for this job">*</span>}
      </dt>
      <dd className="text-right">
        <span className={`block text-sm font-medium ${value ? 'text-slate-900' : 'text-slate-400'}`}>
          {value ?? (required ? 'needed' : '—')}
        </span>
        {badge && value != null && (
          <span className={`mt-0.5 inline-block rounded px-1.5 text-[10px] font-semibold ${badge.cls}`} title={slot?.basis}>
            {badge.label}
          </span>
        )}
      </dd>
    </div>
  )
}

const READINESS_STYLE: Record<string, { label: string; cls: string }> = {
  READY: { label: 'Ready — manual search allowed', cls: 'bg-emerald-600' },
  NEEDS_INFORMATION: { label: 'Collecting details', cls: 'bg-sky-600' },
  NEEDS_CONFIRMATION: { label: 'Awaiting confirmation', cls: 'bg-amber-500' },
  SAFETY_ESCALATION: { label: 'Safety escalation', cls: 'bg-red-600' },
  OUT_OF_SCOPE: { label: 'Outside manual scope', cls: 'bg-slate-500' },
}

export function VehiclePanel({
  vehicle,
  repair,
  readiness,
  activeProcedure,
  modelYears,
  engines,
  onSave,
}: {
  vehicle: Record<string, SlotView>
  repair: Record<string, SlotView>
  readiness: Readiness | null
  activeProcedure: string | null
  modelYears: number[]
  engines: string[]
  onSave: (v: VehicleSelection) => Promise<void>
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState<VehicleSelection>({})
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const required = new Set([...(readiness?.required_fields ?? []), ...(readiness?.missing_fields ?? [])])
  const rs = readiness?.status ? READINESS_STYLE[readiness.status] : null

  const startEdit = () => {
    setDraft({
      model_year: (vehicle.model_year?.value as number) ?? undefined,
      engine: (vehicle.engine?.value as string) ?? undefined,
      transmission: (vehicle.transmission?.value as VehicleSelection['transmission']) ?? undefined,
      body_style: (vehicle.body_style?.value as VehicleSelection['body_style']) ?? undefined,
    })
    setErr(null)
    setEditing(true)
  }
  const save = async () => {
    setSaving(true)
    setErr(null)
    try {
      const clean = Object.fromEntries(Object.entries(draft).filter(([, v]) => v !== undefined && v !== '')) as VehicleSelection
      await onSave(clean)
      setEditing(false)
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Could not update the vehicle.')
    } finally {
      setSaving(false)
    }
  }

  const select = (field: keyof VehicleSelection, label: string, options: (string | number)[]) => (
    <label className="block text-sm">
      <span className="text-slate-500">{label}</span>
      <select
        className="mt-0.5 block w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-slate-900"
        value={draft[field] ?? ''}
        onChange={(e) =>
          setDraft((d) => ({ ...d, [field]: field === 'model_year' ? (e.target.value ? Number(e.target.value) : undefined) : e.target.value || undefined }))
        }
      >
        <option value="">—</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  )

  return (
    <aside className="space-y-3" aria-label="Vehicle and task context">
      <section className="rounded-xl border border-slate-200 bg-white p-3.5 shadow-sm">
        <div className="mb-1 flex items-center gap-2">
          <Car className="h-4 w-4 text-slate-500" aria-hidden />
          <h2 className="text-sm font-semibold text-slate-900">Vehicle</h2>
          {!editing ? (
            <button onClick={startEdit} className="ml-auto inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-sky-700 hover:bg-sky-50">
              <Pencil className="h-3.5 w-3.5" aria-hidden /> Edit
            </button>
          ) : (
            <button onClick={() => setEditing(false)} className="ml-auto rounded p-1 text-slate-500 hover:bg-slate-100" aria-label="Cancel editing">
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
        {editing ? (
          <div className="space-y-2">
            {select('model_year', 'Model year', modelYears)}
            {select('engine', 'Engine', engines)}
            {select('transmission', 'Transmission', ['automatic', 'manual'])}
            {select('body_style', 'Body style', ['coupe', 'convertible'])}
            {err && <p className="text-xs text-red-700">{err}</p>}
            <button
              onClick={save}
              disabled={saving}
              className="w-full rounded-md bg-slate-800 py-1.5 text-sm font-medium text-white hover:bg-slate-900 disabled:opacity-60"
            >
              {saving ? 'Saving…' : 'Save and re-check readiness'}
            </button>
            <p className="text-[11px] text-slate-500">Changing the vehicle invalidates previous answers for a different configuration.</p>
          </div>
        ) : (
          <dl className="divide-y divide-slate-100">
            {VEHICLE_FIELDS.filter(([k]) => k !== 'variant' || vehicle[k]).map(([k, label]) => (
              <Row key={k} label={label} slot={vehicle[k]} required={required.has(k)} />
            ))}
          </dl>
        )}
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-3.5 shadow-sm">
        <div className="mb-1 flex items-center gap-2">
          <Wrench className="h-4 w-4 text-slate-500" aria-hidden />
          <h2 className="text-sm font-semibold text-slate-900">Task</h2>
        </div>
        <dl className="divide-y divide-slate-100">
          {REPAIR_FIELDS.filter(([k]) => repair[k] || k === 'component' || k === 'task_type').map(([k, label]) => (
            <Row key={k} label={label} slot={repair[k]} required={required.has(k)} />
          ))}
        </dl>
        {activeProcedure && (
          <p className="mt-2 rounded bg-slate-50 px-2 py-1 text-xs text-slate-600">
            Current procedure: <span className="font-medium text-slate-800">{activeProcedure}</span>
          </p>
        )}
      </section>

      {readiness && rs && (
        <section className="rounded-xl border border-slate-200 bg-white p-3.5 shadow-sm" aria-label="Readiness">
          <div className="flex items-center gap-2">
            <span className={`h-2.5 w-2.5 rounded-full ${rs.cls}`} aria-hidden />
            <span className="text-sm font-semibold text-slate-900">{rs.label}</span>
          </div>
          <div className="mt-2 h-1.5 overflow-hidden rounded bg-slate-100" role="progressbar" aria-valuenow={Math.round(readiness.score * 100)} aria-valuemin={0} aria-valuemax={100}>
            <div className={`h-full ${rs.cls}`} style={{ width: `${Math.round(readiness.score * 100)}%` }} />
          </div>
          {readiness.missing_fields.length > 0 && (
            <p className="mt-2 text-xs text-slate-600">
              Still needed: {readiness.missing_fields.map((f) => f.replace(/_/g, ' ')).join(', ')}
            </p>
          )}
        </section>
      )}
    </aside>
  )
}

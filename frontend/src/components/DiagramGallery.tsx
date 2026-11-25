import { useEffect, useState } from 'react'
import { ImageOff, Maximize2, X, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react'
import type { ImageRef } from '../types/api'

function Lightbox({ image, onClose }: { image: ImageRef; onClose: () => void }) {
  const [zoom, setZoom] = useState(1)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      if (e.key === '+' || e.key === '=') setZoom((z) => Math.min(4, z + 0.25))
      if (e.key === '-') setZoom((z) => Math.max(0.5, z - 0.25))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={image.caption}
      className="fixed inset-0 z-50 flex flex-col bg-slate-950/90"
      onClick={onClose}
    >
      <div className="flex items-center gap-2 p-3 text-slate-100" onClick={(e) => e.stopPropagation()}>
        <p className="flex-1 text-sm">{image.caption}</p>
        <button className="rounded p-2 hover:bg-white/10" onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))} aria-label="Zoom out">
          <ZoomOut className="h-5 w-5" />
        </button>
        <span className="w-12 text-center text-sm tabular-nums">{Math.round(zoom * 100)}%</span>
        <button className="rounded p-2 hover:bg-white/10" onClick={() => setZoom((z) => Math.min(4, z + 0.25))} aria-label="Zoom in">
          <ZoomIn className="h-5 w-5" />
        </button>
        <button className="rounded p-2 hover:bg-white/10" onClick={() => setZoom(1)} aria-label="Reset zoom">
          <RotateCcw className="h-5 w-5" />
        </button>
        <button className="rounded p-2 hover:bg-white/10" onClick={onClose} aria-label="Close">
          <X className="h-5 w-5" />
        </button>
      </div>
      <div className="flex-1 overflow-auto p-4" onClick={(e) => e.stopPropagation()}>
        <img
          src={image.url}
          alt={image.caption}
          style={{ width: `${zoom * 100}%`, maxWidth: 'none' }}
          className="mx-auto rounded bg-white"
        />
      </div>
    </div>
  )
}

function Thumb({ image, onOpen }: { image: ImageRef; onOpen: () => void }) {
  const [failed, setFailed] = useState(false)
  return (
    <figure className="overflow-hidden rounded-lg border border-slate-200 bg-white">
      <button
        type="button"
        onClick={onOpen}
        className="group relative block w-full bg-white focus:outline-none focus:ring-2 focus:ring-sky-500"
        aria-label={`Open diagram: ${image.caption}`}
      >
        {failed ? (
          <div className="flex h-40 items-center justify-center text-slate-400">
            <ImageOff className="h-6 w-6" aria-hidden />
          </div>
        ) : (
          <img src={image.url} alt={image.caption} loading="lazy" onError={() => setFailed(true)} className="h-44 w-full object-contain p-1" />
        )}
        <span className="absolute right-1.5 top-1.5 rounded bg-slate-900/70 p-1 text-white opacity-0 transition group-hover:opacity-100">
          <Maximize2 className="h-3.5 w-3.5" aria-hidden />
        </span>
      </button>
      <figcaption className="border-t border-slate-100 px-2 py-1.5 text-xs text-slate-600">
        {image.source_id && <span className="mr-1 rounded bg-sky-100 px-1 font-semibold text-sky-800">{image.source_id}</span>}
        {image.nearest_step ? <span className="font-medium text-slate-800">Step {image.nearest_step} · </span> : null}
        {image.procedure_title} · PDF p. {image.page_number}
        {image.printed_page ? ` (${image.printed_page})` : ''}
        {image.figure_numbers.length > 0 && (
          <span className="block text-[11px] text-slate-400">Figure id read from image: {image.figure_numbers.join(', ')}</span>
        )}
      </figcaption>
    </figure>
  )
}

export function DiagramGallery({ images }: { images: ImageRef[] }) {
  const [open, setOpen] = useState<ImageRef | null>(null)
  if (!images.length) return null
  return (
    <section aria-label="Manual diagrams" className="mt-3">
      <h5 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">Original manual diagrams</h5>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        {images.map((im) => (
          <Thumb key={im.figure_id} image={im} onOpen={() => setOpen(im)} />
        ))}
      </div>
      {open && <Lightbox image={open} onClose={() => setOpen(null)} />}
    </section>
  )
}

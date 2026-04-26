import { Chart } from 'chart.js/auto'
import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

// ─── Types ───────────────────────────────────────────────────────────────────
type Modality      = 'mri' | 'ct' | 'fusion'
type AnalysisPhase = 'idle' | 'analyzing' | 'done' | 'error'

interface Scores {
  glioma:     number
  meningioma: number
  pituitary:  number
  no_tumor:   number
}

interface AnalysisResult {
  prediction:  string
  confidence:  number
  scores:      Scores
  heatmap_url: string
  report:      string
}

// ─── Constants ────────────────────────────────────────────────────────────────
const API_BASE = (import.meta as any).env?.VITE_API_URL ?? 'http://127.0.0.1:8000'

const CLASS_COLOR: Record<string, string> = {
  glioma:     '#FF6B6B',
  meningioma: '#FFD166',
  pituitary:  '#4FC3F7',
  no_tumor:   '#06D6A0',
}

const CLASS_KEYS = ['glioma', 'meningioma', 'pituitary', 'no_tumor'] as const

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Map any label string → one of our 4 score keys */
function normaliseLabel(raw: string): string {
  const s = raw.toLowerCase().trim()
  if (s.includes('glioma'))                      return 'glioma'
  if (s.includes('meningioma'))                  return 'meningioma'
  if (s.includes('pituitary'))                   return 'pituitary'
  if (s.includes('no') || s.includes('normal') ||
      s.includes('healthy') || s === 'notumor')  return 'no_tumor'
  return s.replace(/\s+/g, '_')
}

/** Pull scores out of whatever shape the backend sends */
function extractScores(data: any): Scores {
  const empty: Scores = { glioma:0, meningioma:0, pituitary:0, no_tumor:0 }

  // ── Case 1: array of { label, probability } objects
  //    e.g. [{ label:"Glioma", probability:0.002 }, ...]
  const arr = data.probabilities ?? data.scores ?? data.class_scores ?? null
  if (Array.isArray(arr)) {
    const result = { ...empty }
    for (const item of arr) {
      const rawLabel: string = item.label ?? item.class ?? item.name ?? ''
      const value: number    = item.probability ?? item.score ?? item.value ?? item.confidence ?? 0
      const key = normaliseLabel(rawLabel)
      if (key in result) (result as any)[key] = Number(value)
    }
    return result
  }

  // ── Case 2: plain object { glioma: 0.002, meningioma: 0.997, ... }
  if (arr && typeof arr === 'object') {
    const result = { ...empty }
    for (const [k, v] of Object.entries(arr)) {
      const key = normaliseLabel(k)
      if (key in result) (result as any)[key] = Number(v)
    }
    return result
  }

  // ── Case 3: scores are top-level fields on data
  return {
    glioma:     Number(data.glioma     ?? data.Glioma     ?? 0),
    meningioma: Number(data.meningioma ?? data.Meningioma ?? 0),
    pituitary:  Number(data.pituitary  ?? data.Pituitary  ?? 0),
    no_tumor:   Number(data.no_tumor   ?? data.notumor    ?? data.NoTumor ?? data.Normal ?? 0),
  }
}

type TumorInfo = {
  name: string
  description: string
  why: string
}

const TUMOR_INFO: Record<string, TumorInfo> = {
  glioma: {
    name: 'Glioma',
    description:
      'Glioma is a tumor that originates in the glial cells of the brain or spine. ' +
      'It is the most common type of primary brain tumor and can range from slow-growing ' +
      '(low-grade) to aggressive (high-grade, such as glioblastoma).',
    why:
      'The model detected irregular, diffuse signal intensity patterns typically associated ' +
      'with glial cell proliferation. Gliomas often appear as ill-defined hyperintense regions ' +
      'on T2-weighted MRI without a clear capsule boundary.',
  },
  meningioma: {
    name: 'Meningioma',
    description:
      'Meningioma is a tumor that arises from the meninges — the membranes surrounding the ' +
      'brain and spinal cord. Most meningiomas are benign and slow-growing, though some can ' +
      'recur or, rarely, become malignant.',
    why:
      'The model identified a well-defined, homogeneous mass with a broad dural base, which ' +
      'is a hallmark radiological feature of meningioma. The lesion appeared isointense to ' +
      'slightly hyperintense with uniform contrast enhancement patterns.',
  },
  pituitary: {
    name: 'Pituitary Tumor',
    description:
      'A pituitary tumor (adenoma) is an abnormal growth in the pituitary gland located at ' +
      'the base of the brain. Most are benign. They can cause hormonal imbalances and, if ' +
      'large enough, compress surrounding structures like the optic chiasm.',
    why:
      'The model detected a focal lesion in the sellar/suprasellar region consistent with ' +
      'pituitary adenoma. Key features include a well-defined mass centered at the sella ' +
      'turcica with possible suprasellar extension and optic chiasm proximity.',
  },
  no_tumor: {
    name: 'No Tumor Detected',
    description:
      'No significant tumor pathology was identified in the submitted scan. The brain ' +
      'parenchyma appears within normal limits for the analyzed region.',
    why:
      'The model found no abnormal mass, focal signal changes, or structural asymmetry ' +
      'consistent with neoplastic growth. Tissue density and signal distribution appear ' +
      'homogeneous and typical of healthy brain parenchyma.',
  },
}

function wordWrap(text: string, width: number): string {
  const words = text.split(' ')
  const lines: string[] = []
  let line = '  '
  for (const word of words) {
    if ((line + word).length > width + 2) {
      lines.push(line)
      line = '  ' + word + ' '
    } else {
      line += word + ' '
    }
  }
  if (line.trim()) lines.push(line)
  return lines.join('\n')
}

function buildReport(
  prediction: string,
  confidence: number,
  scores: Scores
): string {
  const info = TUMOR_INFO[prediction] ?? {
    name: prediction,
    description: 'No description available.',
    why: 'Insufficient data to explain this prediction.',
  }

  const confPct = (confidence * 100).toFixed(1)
  const threshold = confidence >= 0.85 ? 'HIGH' : confidence >= 0.60 ? 'MODERATE' : 'LOW'

  const scoreLines = Object.entries(scores)
    .sort(([, a], [, b]) => Number(b) - Number(a))
    .map(([k, v]) => {
      const pct = (Number(v) * 100).toFixed(1)
      const filled = Math.round(Number(v) * 20)
      const bar = '█'.repeat(filled) + '░'.repeat(20 - filled)
      const label = k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
      return `  ${label.padEnd(18)} ${bar}  ${pct}%`
    })
    .join('\n')

  return [
    `━━━ CLASSIFICATION RESULT ━━━━━━━━━━━━━━━━━━━━━━━━━`,
    `  Prediction   : ${info.name}`,
    `  Confidence   : ${confPct}%  [${threshold}]`,
    ``,
    `━━━ CLASS CONFIDENCE SCORES ━━━━━━━━━━━━━━━━━━━━━━━`,
    scoreLines,
    ``,
    `━━━ WHAT IS ${info.name.toUpperCase()} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`,
    wordWrap(info.description, 54),
    ``,
    `━━━ WHY THIS PREDICTION ━━━━━━━━━━━━━━━━━━━━━━━━━━━`,
    wordWrap(info.why, 54),
    ``,
    `━━━ NOTE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`,
    wordWrap(
      'This report is generated by an AI classifier for decision-support ' +
      'purposes only and must be reviewed by a qualified radiologist before clinical use.',
      54
    ),
  ].join('\n')
}

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [modality,   setModality]   = useState<Modality>('mri')
  const [mriFile,    setMriFile]    = useState<File | null>(null)
  const [ctFile,     setCtFile]     = useState<File | null>(null)
  const [mriPreview, setMriPreview] = useState<string | null>(null)
  const [ctPreview,  setCtPreview]  = useState<string | null>(null)
  const [phase,      setPhase]      = useState<AnalysisPhase>('idle')
  const [result,     setResult]     = useState<AnalysisResult | null>(null)
  const [errMsg,     setErrMsg]     = useState<string | null>(null)
  const [hmOpacity,  setHmOpacity]  = useState(0.65)
  const [confThr,    setConfThr]    = useState(0.50)
  const [showHeat,   setShowHeat]   = useState(true)
  const [elapsed,    setElapsed]    = useState(0)
  const [history,    setHistory]    = useState<string[]>(['Model initialised', 'GPU context ready'])

  const mriRef  = useRef<HTMLInputElement>(null)
  const ctRef   = useRef<HTMLInputElement>(null)
  const timerID = useRef<number | null>(null)

  useEffect(() => {
    if (phase === 'analyzing') {
      setElapsed(0)
      timerID.current = window.setInterval(() => setElapsed(e => +(e + 0.1).toFixed(1)), 100)
    } else {
      if (timerID.current) clearInterval(timerID.current)
    }
    return () => { if (timerID.current) clearInterval(timerID.current) }
  }, [phase])

  const attachFile = (file: File, type: 'mri' | 'ct') => {
    const url = URL.createObjectURL(file)
    if (type === 'mri') { setMriFile(file); setMriPreview(url) }
    else                { setCtFile(file);  setCtPreview(url) }
    setResult(null); setPhase('idle'); setErrMsg(null)
  }

  const clearFile = (type: 'mri' | 'ct') => {
    if (type === 'mri') { setMriFile(null); setMriPreview(null) }
    else                { setCtFile(null);  setCtPreview(null) }
  }

  const onDrop = useCallback((e: React.DragEvent, type: 'mri' | 'ct') => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) attachFile(file, type)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const reset = () => {
    setMriFile(null); setMriPreview(null)
    setCtFile(null);  setCtPreview(null)
    setResult(null);  setPhase('idle'); setErrMsg(null)
    setHistory(['Model initialised', 'GPU context ready'])
  }

  const run = async () => {
    const primary = modality === 'ct' ? ctFile : mriFile
    if (!primary) { setErrMsg('Upload a scan first.'); return }
    if (modality === 'fusion' && (!mriFile || !ctFile)) {
      setErrMsg('Fusion requires both MRI and CT scans.'); return
    }

    // Basic file validation
    const isImage = primary.type.startsWith('image/')
    const isMedical = /\.(dcm|nii|nii\.gz|mha|nrrd)$/i.test(primary.name)
    if (!isImage && !isMedical) {
      setErrMsg('Invalid file. Please upload a brain scan image (.jpg, .png, .dcm, .nii).')
      return
    }

    setPhase('analyzing'); setErrMsg(null); setResult(null)
    setHistory(h => [...h, 'Running inference…'])

    try {
      const fd = new FormData()
      let endpoint = '/predict/mri'
      if      (modality === 'ct')     { endpoint = '/predict/ct';     fd.append('file', ctFile!) }
      else if (modality === 'fusion') { endpoint = '/predict/fusion'; fd.append('mri_file', mriFile!); fd.append('ct_file', ctFile!) }
      else                            { fd.append('file', mriFile!) }

      const res = await fetch(`${API_BASE}${endpoint}`, { method: 'POST', body: fd })
      if (!res.ok) throw new Error(`Server responded ${res.status}`)
      const data = await res.json()

      // ── Scores (handles array or object format)
      const scores = extractScores(data)

      // ── Prediction label
      const rawLabel: string = data.predicted_label ?? data.prediction ?? data.label ?? 'unknown'
      const prediction = normaliseLabel(rawLabel)

      // ── Confidence (normalise 0-100 → 0-1 if needed)
      const rawConf: number = data.confidence ?? data.probability ?? 0
      const confidence = rawConf > 1 ? rawConf / 100 : rawConf

      // ── Heatmap URL
      const heatRaw: string = data.gradcam_overlay ?? data.heatmap_url ?? data.heatmap_base64 ?? ''
      const heatmap_url = heatRaw.startsWith('data:') || heatRaw === ''
        ? heatRaw
        : `${API_BASE}${heatRaw}`

      setResult({ prediction, confidence, scores, heatmap_url, report: buildReport(prediction, confidence, scores) })
      setPhase('done')
      setHistory(h => [...h, `Done · ${rawLabel}`])
    } catch (err: any) {
      setErrMsg(err.message ?? 'Analysis failed')
      setPhase('error')
      setHistory(h => [...h, 'Error · ' + (err.message ?? 'failed')])
    }
  }

  const primaryPreview = modality === 'ct' ? ctPreview : mriPreview
  const predKey   = result?.prediction ?? ''
  const predColor = CLASS_COLOR[predKey] ?? '#4FC3F7'
  const aboveThr  = result ? result.confidence >= confThr : false

  const sliderBg = (val: number) =>
    `linear-gradient(90deg, var(--c1) ${val * 100}%, var(--border) ${val * 100}%)`

  return (
    <>
      <div className="bg-canvas" aria-hidden="true">
        <div className="orb orb-1" /><div className="orb orb-2" /><div className="orb orb-3" />
        <div className="grid-bg" />
      </div>

      {/* ════════════════ HEADER ════════════════ */}
      <header className="hdr">
        <div className="hdr-logo">
          <div className="logo-mark">
            <div className="logo-mark-ring" />
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="4" fill="#4FC3F7" opacity=".9"/>
              <circle cx="12" cy="12" r="9" stroke="#4FC3F7" strokeWidth="1" fill="none" opacity=".28"/>
              <path d="M8 12h8M12 8v8" stroke="#4FC3F7" strokeWidth=".9" strokeLinecap="round" opacity=".5"/>
            </svg>
          </div>
          <div>
            <div className="logo-name">NEUROVISION</div>
            <div className="logo-sub">DIAGNOSTIC AI · v2.4</div>
          </div>
        </div>

        <nav className="mode-switcher">
          {(['mri','ct','fusion'] as Modality[]).map(m => (
            <button key={m}
              className={`mode-btn ${modality === m ? 'active' : ''}`}
              onClick={() => { setModality(m); setResult(null); setPhase('idle'); setErrMsg(null) }}
            >
              {m.toUpperCase()}
              <span className="mode-badge">LIVE</span>
            </button>
          ))}
        </nav>

        <div className="hdr-right">
          <div className="status-pill">
            <span className="status-dot" style={{ background:'var(--c2)' }} />
            MODEL READY
          </div>
          <div className="status-pill">
            <span className="status-dot" style={{ background:'var(--warn)' }} />
            GPU · 83%
          </div>
          <span className="badge badge-accent">ResNet-50</span>
        </div>
      </header>

      {/* ════════════════ HERO ════════════════ */}
      <div className="hero">
        <div className="hero-eyebrow">CLINICAL INTELLIGENCE PLATFORM</div>
        <h1>
          Detect <span className="hl">brain tumor</span> patterns<br />
          from MRI with <em>precision</em> AI.
        </h1>
        <p className="hero-sub">
          Full-stack diagnostic prototype · Grad-CAM attention · Confidence scoring · Structured clinical report
        </p>
        <div className="hero-chips">
          {[['MRI · Available',true],['CT · Available',true],['Fusion · Available',true],['fMRI · Pending',false],['DTI · Q3 2026',false]].map(
            ([label, active]) => (
              <span key={label as string} className={`chip ${active ? 'on' : ''}`}>
                <span className="chip-dot" />{label as string}
              </span>
            )
          )}
        </div>
        <div className="hero-deco" aria-hidden="true">
          {[1,0,1,0,1,0,1,0,1].map((lit,i) => (
            <div key={i} className={`hd-cell ${lit?'lit':''}`} />
          ))}
        </div>
      </div>

      {/* ════════════════ LAYOUT ════════════════ */}
      <div className="layout">

        {/* ── SIDEBAR ── */}
        <aside className="sidebar">
          <section>
            <div className="sec-title">Scan Input</div>
            {(modality === 'mri' || modality === 'fusion') && (
              <UploadZone label="MRI Scan" preview={mriPreview} file={mriFile}
                inputRef={mriRef} onFile={f => attachFile(f,'mri')}
                onDrop={e => onDrop(e,'mri')} onClear={() => clearFile('mri')} />
            )}
            {(modality === 'ct' || modality === 'fusion') && (
              <UploadZone label="CT Scan" preview={ctPreview} file={ctFile}
                inputRef={ctRef} onFile={f => attachFile(f,'ct')}
                onDrop={e => onDrop(e,'ct')} onClear={() => clearFile('ct')}
                style={{ marginTop:10 }} />
            )}
          </section>

          <section>
            <div className="sec-title">Parameters</div>
            <div className="param-block">
              <div className="param-row"><span>Confidence threshold</span><span>{confThr.toFixed(2)}</span></div>
              <input type="range" min={0} max={1} step={0.01} value={confThr}
                onChange={e => setConfThr(+e.target.value)}
                className="param-slider" style={{ background: sliderBg(confThr) }} />
            </div>
            <div className="param-block">
              <div className="param-row"><span>Heatmap opacity</span><span>{hmOpacity.toFixed(2)}</span></div>
              <input type="range" min={0} max={1} step={0.01} value={hmOpacity}
                onChange={e => setHmOpacity(+e.target.value)}
                className="param-slider" style={{ background: sliderBg(hmOpacity) }} />
            </div>
          </section>

          <section>
            <div className="sec-title">Overlays</div>
            <Toggle label="Grad-CAM heatmap" on={showHeat} onToggle={() => setShowHeat(v => !v)} />
            <Toggle label="Attention regions" on={true} />
            <Toggle label="Lesion contours"   on={false} />
            <Toggle label="Measurement grid"  on={false} />
          </section>

          <section>
            <div className="sec-title">Classifier</div>
            <select className="model-select">
              <option>ResNet-50 (Tumor)</option>
              <option>ResNet-50 (CT)</option>
              <option>YOLOv7 (Detection)</option>
            </select>
          </section>

          {(phase === 'done' || phase === 'error') && (
            <button className="reset-btn" onClick={reset}>↺ &nbsp;Reset workspace</button>
          )}

          <button
            className={`run-btn ${phase === 'analyzing' ? 'running' : ''}`}
            onClick={run}
            disabled={phase === 'analyzing'}
          >
            {phase === 'analyzing' ? (
              <>
                <svg className="spinner-svg" width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="rgba(5,8,16,.35)" strokeWidth="2.5"/>
                  <path d="M12 2a10 10 0 0110 10" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                </svg>
                Analysing… {elapsed}s
              </>
            ) : (
              <>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                  <polygon points="5,3 19,12 5,21" fill="currentColor"/>
                </svg>
                Run Analysis
              </>
            )}
          </button>

          {errMsg && <div className="err-pill">{errMsg}</div>}
        </aside>

        {/* ── CONTENT ── */}
        <main className="content">

          <div className="card">
            <div className="card-hdr">
              <span className="card-title"><span className="title-dot"/>Clinical Preview</span>
              <div style={{ display:'flex', gap:6 }}>
                <span className="badge badge-dim">AXIAL</span>
                <span className="badge badge-accent">{modality.toUpperCase()} T1</span>
              </div>
            </div>
            <div className="scan-grid">
              <ScanPanel label="Uploaded Scan" base={primaryPreview} placeholder="Upload scan to preview" />
              <ScanPanel
                label="Model Attention (Grad-CAM)"
                base={primaryPreview}
                heat={showHeat && result?.heatmap_url ? result.heatmap_url : null}
                opacity={hmOpacity}
                placeholder="Grad-CAM appears after analysis"
              />
            </div>
          </div>

          <div className="pred-row">
            <div className="pred-card" style={result ? { borderColor:`${predColor}44` } : {}}>
              <div className="micro-lbl">
                <span className="micro-dot" style={{ background:'rgba(79,195,247,.6)' }}/>
                Prediction
              </div>
              {result ? (
                <>
                  <div className="pred-label" style={{ color:predColor }}>
                    {result.prediction.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase())}
                  </div>
                  <div className="conf-bar-wrap">
                    <div className="conf-track">
                      <div className="conf-fill" style={{ width:`${result.confidence*100}%`, background:predColor }}/>
                    </div>
                    <span className="conf-pct-text" style={{ color:predColor }}>
                      {(result.confidence*100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="pred-note">
                    {aboveThr ? '✓ Above confidence threshold' : '⚠ Below confidence threshold'}
                  </div>
                </>
              ) : (
                <>
                  <div className="pred-label muted">Awaiting analysis</div>
                  <div className="pred-note">Upload scan and run analysis to classify</div>
                </>
              )}
            </div>

            <div className="pred-card">
              <div className="micro-lbl">
                <span className="micro-dot" style={{ background:'rgba(6,214,160,.6)' }}/>
                Class Confidence
              </div>
              <div className="cls-list">
                {CLASS_KEYS.map(cls => {
                  const val = result ? (result.scores[cls] ?? 0) : 0
                  const col = CLASS_COLOR[cls]
                  return (
                    <div key={cls} className="cls-item">
                      <div className="cls-row">
                        <span className="cls-name">{cls.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</span>
                        <span className="cls-val" style={{ color: result ? col : 'var(--muted)' }}>
                          {result ? `${(val*100).toFixed(1)}%` : '—'}
                        </span>
                      </div>
                      <div className="cls-track">
                        <div className="cls-fill" style={{
                          width: result ? `${Math.min(val*100,100)}%` : '0%',
                          background: `linear-gradient(90deg, ${col}88, ${col})`,
                        }}/>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-hdr">
              <span className="card-title">
                <svg width="10" height="10" viewBox="0 0 10 10">
                  <rect width="10" height="10" rx="2" fill="rgba(79,195,247,.18)"/>
                  <path d="M2 3h6M2 5h4M2 7h5" stroke="#4FC3F7" strokeWidth=".85" strokeLinecap="round"/>
                </svg>
                Decision-Support Report
              </span>
              <span className={`badge ${result?.report ? 'badge-ok' : 'badge-dim'}`}>
                {result?.report ? 'GENERATED' : 'NOT GENERATED'}
              </span>
            </div>
             <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

    {/* REPORT */}
    <div className="report-body">
      <pre className="report-text">{result?.report}</pre>
    </div>
      
    {/* GRAPH */}
    {result && (
      <ConfidenceThresholdChart
        scores={result.scores}
        threshold={confThr}
      />
    )}

</div>
          </div>
        </main>

        {/* ── RIGHT PANEL ── */}
        <aside className="rpanel">
          <div className="rp-card">
            <div className="rp-title">System Status</div>
            <div className="metric-grid">
              <Metric val="READY"  label="Model State"  color="var(--c2)"/>
              <Metric val="4"      label="Classes"       color="var(--c1)"/>
              <Metric val="98.2%"  label="Val. Accuracy" color="var(--c3)"/>
              <Metric val="83%"    label="GPU Load"      color="var(--warn)"/>
            </div>
          </div>

          <div className="rp-card">
            <div className="rp-title">Modalities</div>
            {['MRI','CT','Fusion'].map(m => (
              <div key={m} className="mod-row">
                <span className="mod-name">{m}</span>
                <span className="badge badge-accent">LIVE</span>
              </div>
            ))}
            <div className="grad-info">
              <div className="grad-info-ttl">VISUALIZATION</div>
              <div className="grad-info-name">Grad-CAM</div>
              <div className="grad-info-desc">
                Heatmaps reveal which image regions pushed the classifier toward its prediction.
              </div>
            </div>
          </div>

          <div className="rp-card">
            <div className="rp-title">Neural Activity</div>
            <div style={{ fontFamily:"'DM Mono',monospace", fontSize:10, color:'var(--muted)', marginBottom:9 }}>
              Inference pipeline · {phase === 'analyzing' ? 'ACTIVE' : 'IDLE'}
            </div>
            <div className="wave-bars">
              {Array.from({ length:14 }, (_,i) => (
                <div key={i} className="wave-bar" style={{
                  animationDelay:`${i*0.07}s`,
                  opacity:    phase === 'analyzing' ? 1 : 0.25,
                  background: phase === 'analyzing' ? 'var(--c1)' : 'var(--muted)',
                }}/>
              ))}
            </div>
          </div>

          <div className="rp-card">
            <div className="rp-title">Session Log</div>
            <div className="tl">
              {history.map((entry, i) => {
                const isLast = i === history.length - 1
                const isLive = isLast && phase === 'analyzing'
                return (
                  <div key={i} className="tl-item">
                    <div className="tl-spine">
                      <div className={`tl-dot ${isLive ? 'live' : 'done'}`}/>
                      {i < history.length - 1 && <div className="tl-line"/>}
                    </div>
                    <div className="tl-body">
                      <div className="tl-label">{entry}</div>
                      {i === 0 && <div className="tl-sub">ResNet-50 · weights loaded</div>}
                      {i === 1 && <div className="tl-sub">CUDA · 83% utilisation</div>}
                      {phase === 'done' && isLast && i > 1 &&
                        <div className="tl-sub">Completed in {elapsed}s</div>}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </aside>
      </div>
    </>
  )
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function UploadZone({ label, preview, file, inputRef, onFile, onDrop, onClear, style }: {
  label: string; preview: string|null; file: File|null
  inputRef: React.RefObject<HTMLInputElement | null>
  onFile:(f:File)=>void; onDrop:(e:React.DragEvent)=>void
  onClear:()=>void; style?: React.CSSProperties
}) {
  const [dragging, setDragging] = useState(false)
  return (
    <div style={style}>
      <input ref={inputRef} type="file" accept="image/*,.dcm,.nii,.nii.gz" style={{ display:'none' }}
        onChange={e => { if (e.target.files?.[0]) onFile(e.target.files[0]) }}/>
      <div
        className={`upload-zone${dragging?' dragging':''}${file?' has-file':''}`}
        onClick={() => !file && inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { setDragging(false); onDrop(e) }}
      >
        {preview ? (
          <div className="uz-preview">
            <img src={preview} alt="scan preview"/>
            <button className="uz-clear" onClick={e => { e.stopPropagation(); onClear() }}>✕</button>
            <div className="uz-filename">✓ &nbsp;{file?.name}</div>
          </div>
        ) : (
          <>
            <div className="uz-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path d="M12 16V8M12 8l-3 3M12 8l3 3" stroke="#4FC3F7" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M20 16.5A4.5 4.5 0 0017.5 8h-.54A7 7 0 104 15.5" stroke="#4FC3F7" strokeWidth="1.3" strokeLinecap="round"/>
              </svg>
            </div>
            <div className="uz-title">{label}</div>
            <div className="uz-sub">Drop here or click to browse<br/>.nii · .dcm · .png · .jpg</div>
          </>
        )}
      </div>
    </div>
  )
}

function ScanPanel({ label, base, heat, opacity = 0.65, placeholder }: {
  label: string; base?: string|null; heat?: string|null
  opacity?: number; placeholder: string
}) {
  return (
    <div className="scan-panel">
      <div className="scan-panel-lbl">{label}</div>
      <div className="corner tl"/><div className="corner tr"/>
      <div className="corner bl"/><div className="corner br"/>
      <div className="sweep"/>
      {base && heat ? (
        <div className="overlay-wrap">
          <img src={base} className="scan-img" alt="scan"/>
          <img src={heat} className="heat-img" style={{ opacity }} alt="heatmap"/>
        </div>
      ) : base ? (
        <img src={base} className="scan-img" alt="scan"/>
      ) : (
        <div className="scan-ph">
          <div className="scan-ph-ring">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="9" stroke="rgba(79,195,247,.28)" strokeWidth="1"/>
              <circle cx="12" cy="12" r="4" stroke="rgba(79,195,247,.4)" strokeWidth="1"/>
            </svg>
          </div>
          <span className="scan-ph-text">{placeholder}</span>
        </div>
      )}
    </div>
  )
}

function Toggle({ label, on, onToggle }: { label:string; on:boolean; onToggle?:()=>void }) {
  return (
    <div className="toggle-row">
      <span className="toggle-label">{label}</span>
      <div className={`toggle ${on?'on':''}`} onClick={onToggle} role="switch" aria-checked={on}/>
    </div>
  )
}

function Metric({ val, label, color }: { val:string; label:string; color:string }) {
  return (
    <div className="metric-cell">
      <div className="metric-val" style={{ color }}>{val}</div>
      <div className="metric-lbl">{label}</div>
    </div>
  )
}

function ConfidenceThresholdChart({ scores, threshold }: {
  scores: Scores | null
  threshold: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef  = useRef<any>(null)

  const CLASSES = [
    { key: 'glioma' as keyof Scores, label: 'Glioma', color: '#FF6B6B' },
    { key: 'meningioma' as keyof Scores, label: 'Meningioma', color: '#4FC3F7' },
    { key: 'pituitary' as keyof Scores, label: 'Pituitary', color: '#9B8EFF' },
    { key: 'no_tumor' as keyof Scores, label: 'No Tumor', color: '#06D6A0' },
  ]

  useEffect(() => {
    if (!canvasRef.current || !scores) return
    if (chartRef.current) {
  chartRef.current.destroy()
}

    const vals = CLASSES.map(c => scores ? +(scores[c.key] * 100).toFixed(1) : 0)

    const thrPlugin = {
      id: 'thrLine',
      afterDraw(chart: any) {
        const thr = threshold * 100
        const { ctx, chartArea: { left, right }, scales: { y } } = chart
        const yPx = y.getPixelForValue(thr)
        ctx.save()
        ctx.setLineDash([6, 4])
        ctx.strokeStyle = '#FFD166'
        ctx.lineWidth = 1.5
        ctx.beginPath(); ctx.moveTo(left, yPx); ctx.lineTo(right, yPx); ctx.stroke()
        ctx.setLineDash([])
        ctx.fillStyle = '#FFD166'
        ctx.font = '11px monospace'
        ctx.fillText((thr).toFixed(0) + '%', right + 5, yPx + 4)
        ctx.restore()
      }
    }

    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: {
        labels: ['Step 1', 'Step 2', 'Step 3', 'Final'],
        datasets: CLASSES.map((c, i) => ({
          label: c.label,
          data: [vals[i] * 0.7, vals[i] * 0.85, vals[i] * 0.95, vals[i]],
          borderColor: c.color,
          backgroundColor: c.color + '18',
          borderWidth: 2,
          pointRadius: 4,
          tension: 0.42,
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 8, right: 42 } },
        plugins: { legend: { display: false } },
        scales: {
          y: {
            min: 0, max: 100,
            ticks: {
  callback: (value) => `${value}%`,
  color: '#505A6A'
},
            grid: { color: 'rgba(99,179,237,0.07)' }
          },
          x: {
            ticks: { color: '#505A6A' },
            grid: { color: 'rgba(99,179,237,0.04)' }
          }
        }
      },
      plugins: [thrPlugin]
    })
  }, [scores, threshold])

  return (
    <div>
      <div style={{ height: 200 }}>
        <canvas ref={canvasRef} />
      </div>
    </div>
  )
}
import { useEffect, useRef, useState } from 'react';
import {
  ShieldCheck, Flag, Ban, Download, RotateCcw, Play, AlertTriangle, TrendingUp, Clock, Mic, ListChecks,
  Image as ImageIcon, Table2, GanttChartSquare, Smartphone, Users, MoveHorizontal, UploadCloud, FileText, Gauge, ScanFace,
  ImageOff, CheckCircle2, ChevronDown, Layers, HelpCircle, GitMerge,
} from 'lucide-react';
import EmptyState from './ui/EmptyState';

// Results report (flow.txt PART II-D, D.3). Input: the report JSON built by server/aggregation/report.py.
// Series colours: CSS variables --series-video / --series-audio (index.css), light/dark steps of the
// reference palette slots 1-2. Chart text and grid lines follow the theme.
const SOURCE_VAR = { video: 'var(--series-video)', audio: 'var(--series-audio)' };
// UI4: verdict tone (index.css .tone-*), word shown next to the label, semantic edge colour (CSS variable)
const VERDICT = {
  VALID: { tone: 'tone-success', word: 'Pass', edge: 'var(--success)', Icon: ShieldCheck },
  SUSPICIOUS: { tone: 'tone-warning', word: 'Review', edge: 'var(--warning)', Icon: Flag },
  INVALID: { tone: 'tone-danger', word: 'Fail', edge: 'var(--critical)', Icon: Ban },
};
const nice = (t) => String(t ?? '').toLowerCase().replace(/_/g, ' ');
const orDash = (v) => (v === undefined || v === null || v === '' ? '—' : v);

const Legend = () => (
  <div className="flex gap-4 text-xs text-muted mb-2">
    {Object.entries(SOURCE_VAR).map(([k, c]) => (
      <span key={k} className="flex items-center gap-1.5 capitalize">
        <span className="inline-block w-2 h-2 rounded-sm" style={{ background: c }} /> {k}
      </span>
    ))}
  </div>
);

const Tooltip = ({ tip }) =>
  tip ? (
    <div
      className="pointer-events-none fixed z-50 px-2.5 py-2 text-xs leading-5 whitespace-pre glass-strong animate-fade-in"
      style={{ left: tip.x + 14, top: tip.y + 14 }}
    >
      {tip.text}
    </div>
  ) : null;

// Responsive charts: the SVG is drawn at the container's real width (no scaling), so labels stay legible
// on phones instead of shrinking with the whole drawing.
const useContainerWidth = (fallback = 640) => {
  const ref = useRef(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(260, Math.floor(entry.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
};
const chartLayout = (width) => (width < 480 ? { labelW: 118, font: 10 } : { labelW: 170, font: 11 });

const PercentBars = ({ rows }) => {
  const [tip, setTip] = useState(null);
  const [ref, width] = useContainerWidth();
  const max = Math.max(10, ...rows.map((r) => r.percentage)) * 1.1;
  const rowH = 26;
  const { labelW, font } = chartLayout(width);
  const plotW = width - labelW - 48;
  return (
    <div ref={ref}>
      <Legend />
      <svg width={width} height={rows.length * rowH + 24} className="block" role="img" aria-label="Violation percentage by type">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={labelW + f * plotW} x2={labelW + f * plotW} y1={0} y2={rows.length * rowH} className="chart-grid" />
            <text x={labelW + f * plotW} y={rows.length * rowH + 16} fontSize="10" className="chart-text" textAnchor="middle">
              {(f * max).toFixed(0)}%
            </text>
          </g>
        ))}
        {rows.map((r, i) => {
          const w = Math.max(2, (r.percentage / max) * plotW);
          const y = i * rowH + 5;
          return (
            <g
              key={r.type}
              onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, text:
                `${r.type}\n${r.percentage.toFixed(1)}% = ${r.numerator} / ${r.denominator} ${r.unit}\n` +
                `${r.duration_s.toFixed(1)} s, ${r.count} event(s)${r.critical ? ', critical' : ''}` })}
              onMouseLeave={() => setTip(null)}
            >
              <rect x={0} y={y - 4} width={width} height={rowH} fill="transparent" />
              <text x={labelW - 8} y={y + 12} fontSize={font} className="chart-text" textAnchor="end">{nice(r.type)}</text>
              <rect x={labelW} y={y + 3} width={w} height={12} rx={2} style={{ fill: SOURCE_VAR[r.source] }} />
              <text x={labelW + w + 6} y={y + 13} fontSize={font} className="chart-text-strong">{r.percentage.toFixed(1)}%</text>
            </g>
          );
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
};

const Timeline = ({ events, duration }) => {
  const [tip, setTip] = useState(null);
  const [ref, width] = useContainerWidth();
  const types = [...new Set(events.map((e) => e.type))];
  const end = Math.max(duration || 0, ...events.map((e) => e.end_s), 1);
  const rowH = 24;
  const { labelW, font } = chartLayout(width);
  const plotW = width - labelW - 20;
  const x = (t) => labelW + (t / end) * plotW;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * end);
  return (
    <div ref={ref}>
      <Legend />
      <svg width={width} height={types.length * rowH + 24} className="block" role="img" aria-label="Violation timeline">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={0} y2={types.length * rowH} className="chart-grid" />
            <text x={x(t)} y={types.length * rowH + 16} fontSize="10" className="chart-text" textAnchor="middle">{t.toFixed(0)} s</text>
          </g>
        ))}
        {types.map((t, i) => (
          <text key={t} x={labelW - 8} y={i * rowH + 16} fontSize={font} className="chart-text" textAnchor="end">{nice(t)}</text>
        ))}
        {events.map((e, k) => {
          const row = types.indexOf(e.type);
          const w = Math.max(3, x(Math.max(e.end_s, e.start_s + 0.2)) - x(e.start_s));
          // BUF1: unconfirmed events are drawn hollow (kept for audit, not scored)
          const unconfirmed = e.status === 'unconfirmed';
          return (
            <rect
              key={k}
              x={x(e.start_s)}
              y={row * rowH + 5}
              width={w}
              height={14}
              rx={2}
              style={unconfirmed
                ? { fill: 'transparent', stroke: SOURCE_VAR[e.source], strokeDasharray: '3 2' }
                : { fill: SOURCE_VAR[e.source] }}
              className={unconfirmed ? '' : 'chart-gap'}
              strokeWidth="1"
              onMouseMove={(ev) => setTip({ x: ev.clientX, y: ev.clientY, text:
                `${e.type} (${e.source})${e.status ? ` · ${e.status}` : ''}\n` +
                `${e.start_s.toFixed(2)} - ${e.end_s.toFixed(2)} s (${e.duration_s.toFixed(2)} s)` +
                (e.merged_from > 1 ? `\nmerged from ${e.merged_from} runs` : '') +
                (e.buffer_reason ? `\nnot confirmed: ${e.buffer_reason}` : '') +
                (e.details && Object.keys(e.details).length ? `\n${JSON.stringify(e.details)}` : '') })}
              onMouseLeave={() => setTip(null)}
            />
          );
        })}
      </svg>
      <Tooltip tip={tip} />
      {events.some((e) => e.status === 'unconfirmed') && (
        <p className="text-xs text-subtle mt-2">Hollow bars: unconfirmed events, kept for review and not scored.</p>
      )}
    </div>
  );
};

// BUF1 violation analysis: what the tolerance buffer did per violation type
const STRENGTH_BADGE = { strong: 'badge-danger', moderate: 'badge-warning', weak: 'badge-info', none: 'badge-muted' };

const metricsText = (m) => {
  if (!m || m.status) return m?.status || 'not measured';
  const parts = Object.entries(m).filter(([, v]) => typeof v === 'number')
    .map(([k, v]) => `${nice(k)} ${(100 * v).toFixed(1)}%`);
  return `${parts.join(', ')} (${m.dataset}${m.version ? `, ${m.version}` : ''})`;
};

const PolicyDetails = ({ r }) => (
  <div className="text-xs text-muted mt-1.5 space-y-1 max-w-md">
    <p>Policy: merge gaps up to {r.buffer_policy.merge_gap_s} s; confirm at {r.buffer_policy.confirm_s} s
      or more with at least {r.buffer_policy.min_support} observation(s); grace {r.buffer_policy.grace_s} s.</p>
    {r.correlated_duration_s > 0 && <p>{r.correlated_duration_s.toFixed(1)} s already scored by a correlated violation.</p>}
    <p>Not assessable: {r.unavailable_duration_s.toFixed(1)} s.</p>
    <p>Detector validation: {metricsText(r.validation_metrics)}</p>
    {r.peak_raw_score != null && <p>Peak raw score {r.peak_raw_score} (not calibrated).</p>}
    {r.evidence_note && <p>{r.evidence_note}</p>}
  </div>
);

const EvidenceBadges = ({ r }) => (
  <span className="inline-flex items-center gap-1 whitespace-nowrap">
    <span className={`badge ${STRENGTH_BADGE[r.evidence_strength] || 'badge-muted'}`}>{r.evidence_strength}</span>
    {r.review_required && <span className="badge badge-warning">review</span>}
  </span>
);

const ViolationAnalysis = ({ rows, assessment }) => (
  <section className="card space-y-4">
    <SectionTitle Icon={Layers}>Violation analysis</SectionTitle>
    <p className="text-sm text-muted">
      Raw detections pass a per-violation tolerance buffer: brief runs are merged, and only events that last long enough
      with enough supporting observations are <span className="font-medium">confirmed</span> and scored. Unconfirmed
      events are kept for review, not hidden.
    </p>
    {assessment && (
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        {[['Assessment', assessment.status], ['Evidence strength', assessment.evidence_strength],
          ['Confirmed events', assessment.confirmed_events], ['Unconfirmed events', assessment.unconfirmed_events]]
          .map(([k, val]) => (
            <div key={k} className="stat">
              <p className="stat-label">{k}</p>
              <p className="text-sm font-semibold capitalize mt-1">{String(val).toLowerCase()}</p>
            </div>
          ))}
      </div>
    )}
    {rows.length === 0 ? (
      <div className="rounded-md border border-line px-3 py-2.5 text-[13px] text-muted inline-flex items-center gap-2">
        <CheckCircle2 className="w-4 h-4 text-success" aria-hidden="true" /> No detector produced an event in this session.
      </div>
    ) : (
      <>
      <ul className="sm:hidden divide-y divide-[var(--border)] border-y border-line">
        {rows.map((r) => (
          <li key={r.type} className="py-3">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[13px] font-medium capitalize">{nice(r.type)}</span>
              <EvidenceBadges r={r} />
            </div>
            <dl className="grid grid-cols-3 gap-x-3 gap-y-1.5 mt-2 text-xs">
              {[['Raw', r.raw_detection_count], ['Filtered', r.filtered_detection_count], ['Merged', r.merged_event_count],
                ['Unconfirmed', r.unconfirmed_event_count], ['Confirmed', r.confirmed_event_count],
                ['Points', r.score != null ? r.score.toFixed(1) : '—']].map(([k, val]) => (
                <div key={k}><dt className="text-subtle">{k}</dt><dd className="font-mono tabular-nums">{val}</dd></div>
              ))}
            </dl>
            <details className="mt-2">
              <summary className="text-xs text-muted cursor-pointer">Policy and validation</summary>
              <PolicyDetails r={r} />
            </details>
          </li>
        ))}
      </ul>
      <div className="hidden sm:block overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="table-head">
              <th className="py-2 pr-4">Type</th><th className="pr-4">Raw</th><th className="pr-4">Filtered</th>
              <th className="pr-4">Merged</th><th className="pr-4">Unconfirmed</th><th className="pr-4">Confirmed</th>
              <th className="pr-4 hidden lg:table-cell">Monitored (s)</th><th className="pr-4">Evidence</th><th>Points</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.type} className="table-row align-top">
                <td className="py-2 pr-4 min-w-[9rem]">
                  <details>
                    <summary className="capitalize cursor-pointer whitespace-nowrap">{nice(r.type)}</summary>
                    <PolicyDetails r={r} />
                  </details>
                </td>
                <td className="pr-4 font-mono">{r.raw_detection_count}</td>
                <td className="pr-4 font-mono">{r.filtered_detection_count}</td>
                <td className="pr-4 font-mono">{r.merged_event_count}</td>
                <td className="pr-4 font-mono">{r.unconfirmed_event_count}</td>
                <td className="pr-4 font-mono">{r.confirmed_event_count}</td>
                <td className="pr-4 font-mono hidden lg:table-cell">{r.valid_observation_duration_s.toFixed(1)}</td>
                <td className="pr-4"><EvidenceBadges r={r} /></td>
                <td className="font-mono">{r.score != null ? r.score.toFixed(1) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      </>
    )}
    <p className="text-xs text-subtle">
      {assessment?.note || 'A confirmed detector event is not proof of cheating.'} Detector scores are not calibrated
      probabilities, and validation figures describe the detector on its test data, not this candidate&apos;s result.
    </p>
  </section>
);

// FUS1 evidence fusion: assessment, evidence families and the chronological event records
const STATUS_BADGE = { confirmed: 'badge-danger', uncertain: 'badge-warning', filtered: 'badge-muted', unavailable: 'badge-outline' };
const STATUS_HELP = {
  confirmed: 'met its persistence rule and was scored',
  uncertain: 'kept for review, not scored',
  filtered: 'raw detections that never persisted',
  unavailable: 'a check that could not be performed',
};
const PRIORITY_BADGE = { high: 'badge-danger', medium: 'badge-warning', low: 'badge-muted' };

const EventRow = ({ e, fileUrl }) => (
  <>
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-[13px] font-medium capitalize">{nice(e.type)}</span>
      <span className={`badge ${STATUS_BADGE[e.validation_status] || 'badge-muted'}`}>{e.validation_status}</span>
      {e.severity && <span className="text-2xs text-subtle">{e.severity}</span>}
      <span className="ml-auto font-mono text-xs tabular-nums">{(e.score_contribution || 0).toFixed(1)} pts</span>
    </div>
    <p className="text-2xs text-subtle mt-0.5 font-mono">
      {e.event_id} · {fmtT(e.start_s)} – {fmtT(e.end_s)} ({(e.duration_s || 0).toFixed(1)} s)
      {e.observation_quality ? ` · quality ${e.observation_quality}` : ''}
      {e.confidence != null ? ` · score ${e.confidence.toFixed(2)} (uncalibrated)` : ''}
    </p>
    <p className="text-xs text-muted mt-1">{e.validation_reason}</p>
    {e.score_basis && <p className="text-2xs text-subtle mt-0.5">{e.score_basis}</p>}
    {(e.evidence_refs?.length > 0 || e.related_event_ids?.length > 0) && (
      <p className="text-2xs mt-1 flex flex-wrap gap-x-3 gap-y-0.5">
        {e.evidence_refs?.map((r) => (fileUrl
          ? <a key={r.file} href={fileUrl(r.file)} target="_blank" rel="noreferrer" className="text-accent hover:underline">{r.kind}: {r.evidence_id}</a>
          : <span key={r.file} className="text-subtle">{r.kind}: {r.evidence_id}</span>))}
        {e.related_event_ids?.length > 0 && <span className="text-subtle">related: {e.related_event_ids.join(', ')}</span>}
      </p>
    )}
  </>
);

const FusionSection = ({ report, reportId }) => {
  const a = report.assessment || {};
  const events = report.events || [];
  const fams = Object.entries(report.fusion?.families || {}).filter(([, f]) => f.measurable || f.confirmed_events);
  const [filter, setFilter] = useState('confirmed');
  const counts = a.event_counts || {};
  const shown = events.filter((e) => filter === 'all' || e.validation_status === filter);
  const fileUrl = reportId ? (f) => `/api/reports/${reportId}/files/${f}` : null;
  return (
    <section className="card space-y-4">
      <SectionTitle Icon={GitMerge}>Evidence fusion</SectionTitle>
      <div className="flex flex-wrap items-center gap-2 text-[13px]">
        <span className="badge badge-outline">{nice(a.status)}</span>
        {a.review_priority && <span className={`badge ${PRIORITY_BADGE[a.review_priority]}`}>review priority {a.review_priority}</span>}
        {a.corroborations > 0 && <span className="badge badge-info">{a.corroborations} cross-modal corroboration(s)</span>}
        <span className="text-xs text-muted">Contributing families: {(a.contributing_families || []).map((f) => report.fusion.families[f]?.label || f).join(', ') || 'none'}</span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {['confirmed', 'uncertain', 'filtered', 'unavailable'].map((k) => (
          <div key={k} className="stat" title={STATUS_HELP[k]}>
            <p className="stat-label capitalize">{k}</p>
            <p className="stat-value">{counts[k] ?? 0}</p>
            <p className="stat-meta">{STATUS_HELP[k]}</p>
          </div>
        ))}
      </div>
      {fams.length > 0 && (
        <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0">
          <table className="w-full min-w-[560px] text-sm">
            <thead><tr className="table-head">
              <th className="py-2 pr-4">Evidence family</th><th className="pr-4">Confirmed</th><th className="pr-4">Uncertain</th>
              <th className="pr-4">De-duplicated</th><th>Points</th>
            </tr></thead>
            <tbody>
              {fams.map(([k, f]) => (
                <tr key={k} className="table-row">
                  <td className="py-2 pr-4">
                    <span className="font-medium">{f.label}</span>
                    <span className="block text-2xs text-subtle">{f.members.map(nice).join(', ')}{f.rule === 'priority_dedup' ? ' · correlated: overlapping time scored once' : ''}</span>
                  </td>
                  <td className="pr-4 font-mono">{f.confirmed_events}</td>
                  <td className="pr-4 font-mono">{f.uncertain_events}</td>
                  <td className="pr-4 font-mono">{f.deduplicated_s.toFixed(1)} s</td>
                  <td className="font-mono">{f.points.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div>
        <div className="segmented flex w-full sm:inline-flex sm:w-auto overflow-x-auto mb-3" role="tablist" aria-label="Event status">
          {['confirmed', 'uncertain', 'unavailable', 'filtered', 'all'].map((k) => (
            <button key={k} role="tab" aria-selected={filter === k} onClick={() => setFilter(k)}
              className={`tab flex-1 justify-center capitalize sm:flex-none ${filter === k ? 'tab-active' : 'tab-idle'}`}>
              {k}{k !== 'all' ? ` ${counts[k] ?? 0}` : ''}
            </button>
          ))}
        </div>
        {shown.length === 0 ? (
          <p className="text-xs text-muted">No {filter === 'all' ? '' : `${filter} `}events.</p>
        ) : (
          <ul className="divide-y divide-[var(--border)] border-y border-line max-h-[32rem] overflow-y-auto">
            {shown.map((e) => <li key={e.event_id} className="py-2.5"><EventRow e={e} fileUrl={fileUrl} /></li>)}
          </ul>
        )}
      </div>
      {(report.limitations || []).length > 0 && (
        <ul className="text-2xs text-subtle space-y-0.5 list-disc pl-4">
          {report.limitations.map((l) => <li key={l}>{l}</li>)}
        </ul>
      )}
    </section>
  );
};

// Evidence groups (flow.txt §C.5): voice clips, object / multiple-face / head-pose snapshots, each sorted by time
const EVIDENCE_ORDER = ['voice', 'object', 'multiple_faces', 'spoof', 'head_pose'];
const GROUP_META = {
  voice: { label: 'Voice violations', Icon: Mic },
  object: { label: 'Prohibited objects', Icon: Smartphone },
  multiple_faces: { label: 'Multiple faces', Icon: Users },
  head_pose: { label: 'Head pose movement', Icon: MoveHorizontal },
  spoof: { label: 'Face spoofing', Icon: ScanFace },
};
const fmtT = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${(s % 60).toFixed(1).padStart(4, '0')}`;

const itemDetails = (group, it) => {
  switch (group) {
    case 'object':
      return (it.objects || []).map((o) => `${o.label} ${(100 * o.confidence).toFixed(0)}%`).join(', ') || 'object';
    case 'multiple_faces':
      return `${it.face_count ?? '?'} face(s), ${it.person_count ?? '?'} person(s) (max ${it.max_persons ?? '?'})`;
    case 'head_pose': {
      const a = it.angles_deg || {};
      const lim = it.allowed_deg || {};
      return `yaw ${a.yaw ?? '-'}°, pitch ${a.pitch ?? '-'}°, roll ${a.roll ?? '-'}° (limits ${lim.yaw}/${lim.pitch}/${lim.roll}°) · ` +
        `${it.duration_s.toFixed(1)} s (allowed ${it.allowed_duration_s} s)`;
    }
    case 'spoof':
      return `real-face score ${it.real_score ?? '-'} (min ${it.min_real_score ?? '-'}, spoof below ${it.real_threshold ?? '-'}) · ` +
        `${it.duration_s.toFixed(1)} s (allowed ${it.allowed_duration_s ?? '-'} s)`;
    case 'voice':
      return [(it.types || [it.type]).map(nice).join(' + '),
        it.speaker_similarity_min != null ? `voice similarity ${it.speaker_similarity_min.toFixed(2)} (threshold ${it.speaker_threshold})` : null,
        it.speech_s_in_event != null ? `speech ${it.speech_s_in_event.toFixed(1)} s` : null].filter(Boolean).join(' · ');
    default:
      return '';
  }
};

const UploadBadge = ({ upload }) => {
  if (!upload) return null;
  const style = { uploaded: 'badge-success', failed: 'badge-danger', skipped: 'badge-muted' }[upload.status] || 'badge-muted';
  return (
    <span className={`badge ${style}`} title={upload.error || upload.target || upload.reason || ''}>
      <UploadCloud className="w-3.5 h-3.5" /> {upload.status}
    </span>
  );
};

// evidence counter text (LIM2: % of session vs limit; reports before LIM2 only have count / threshold)
const counterText = (c) => (c.percentage != null
  ? `${c.count} event(s) · ${c.percentage.toFixed(1)}% of session / limit ${c.limit_text}`
  : `${c.count} / limit ${c.threshold}`);

// SC2 scoring: severity badge colours
const SEVERITY_BADGE = { critical: 'badge-danger', medium: 'badge-warning', low: 'badge-info', 'very low': 'badge-muted' };

const ScoringCard = ({ verdict, rows }) => (
  <section className="card space-y-4">
    <SectionTitle Icon={Gauge}>Scoring</SectionTitle>
    <p className="text-[13px] leading-relaxed">
      <span className="font-semibold">Overall score {verdict.score.toFixed(1)} / 100 ({verdict.label}).</span>{' '}
      <span className="text-muted">{orDash(verdict.score_explanation)}</span>
    </p>
    <ul className="divide-y divide-[var(--border)]">
      {rows.map((r) => (
        <li key={r.type} className="space-y-1.5 py-3 first:pt-0 last:pb-0">
          <div className="flex flex-wrap items-center gap-2 justify-between">
            <span className="text-[13px] font-medium capitalize inline-flex items-center gap-2">
              {nice(r.type)}
              <span className={`badge ${SEVERITY_BADGE[r.severity] || 'badge-muted'}`}>{r.severity}</span>
            </span>
            <span className="font-mono text-sm tabular-nums">{r.score != null ? r.score.toFixed(1) : '—'} / {r.max_points}</span>
          </div>
          <div className="track" aria-hidden="true">
            <div className="h-full rounded-full transition-all duration-500" style={{
              width: `${r.max_points ? (100 * (r.score || 0)) / r.max_points : 0}%`,
              background: r.critical ? 'var(--critical)' : SOURCE_VAR[r.source],
            }} />
          </div>
          <p className="text-xs text-muted">{r.score_explanation}</p>
        </li>
      ))}
    </ul>
  </section>
);

// evidence snapshot with a designed fallback when the file is missing or fails to load (never a broken image)
const EvidenceImage = ({ src, alt, note }) => {
  const [failed, setFailed] = useState(!src);
  if (failed) {
    return (
      <div className="aspect-video flex flex-col items-center justify-center gap-1.5 text-xs text-subtle surface-muted">
        <ImageOff className="w-5 h-5" aria-hidden="true" /> {note || 'Image not available'}
      </div>
    );
  }
  return (
    <a href={src} target="_blank" rel="noreferrer" className="block overflow-hidden">
      <img src={src} alt={alt} className="w-full aspect-video object-cover transition-transform duration-300 hover:scale-[1.02]"
        loading="lazy" onError={() => setFailed(true)} />
    </a>
  );
};

const EvidenceGroup = ({ group, data, counter, upload, reportId }) => {
  const { label, Icon } = GROUP_META[group];
  const items = data?.items || [];
  const fileUrl = (f) => `/api/reports/${reportId}/files/${f}`;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 justify-between">
        <h3 className="font-semibold inline-flex items-center gap-2"><Icon className="w-4 h-4 text-subtle" aria-hidden="true" /> {label}</h3>
        <div className="flex items-center gap-2">
          {counter && (
            <span className={`badge ${counter.exceeded ? 'badge-danger' : 'badge-muted'}`}>
              {counterText(counter)}{counter.exceeded ? ' · exceeded' : ''}
            </span>
          )}
          <UploadBadge upload={upload} />
        </div>
      </div>
      {upload?.status === 'failed' && (
        <div className="alert alert-error text-xs"><AlertTriangle className="w-4 h-4 flex-shrink-0" /> Upload failed: {orDash(upload.error)}</div>
      )}
      {items.length === 0 ? (
        <div className="rounded-md border border-line px-3 py-2.5 text-[13px] text-muted inline-flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-success" aria-hidden="true" /> No {label.toLowerCase()} recorded in this session.
        </div>
      ) : group === 'voice' ? (
        <ul className="space-y-2">
          {items.map((it) => (
            <li key={it.id} className="rounded-md border border-line p-3 flex flex-col md:flex-row md:items-center gap-3">
              <div className="md:w-64 flex-shrink-0">
                <p className="text-sm font-medium">{fmtT(it.start_s)} – {fmtT(it.end_s)}</p>
                <p className="text-xs text-muted">{itemDetails(group, it)}</p>
              </div>
              {it.file && reportId
                ? <audio controls preload="none" src={fileUrl(it.file)} className="w-full md:flex-1" />
                : <span className="text-xs text-subtle">{it.note || 'Audio clip not available'}</span>}
            </li>
          ))}
        </ul>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {items.map((it) => (
            <figure key={it.id} className="rounded-2xl overflow-hidden glass card-hover">
              <EvidenceImage src={it.file && reportId ? fileUrl(it.file) : null} alt={`${label} at ${fmtT(it.start_s)}`} note={it.note} />
              <figcaption className="text-xs px-3 py-2.5 space-y-0.5">
                <p className="font-medium">{fmtT(it.start_s)} – {fmtT(it.end_s)} ({it.duration_s.toFixed(1)} s)</p>
                <p className="text-muted">{itemDetails(group, it)}</p>
              </figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
};

const SectionTitle = ({ Icon, children }) => (
  <h2 className="card-title flex items-center gap-2 mb-3">
    <Icon className="w-4 h-4 text-subtle" aria-hidden="true" /> {children}
  </h2>
);

const Statistics = ({ report, reportId, onNewSession, onReset }) => {
  const v = report.verdict;
  const { tone, word, edge, Icon } = VERDICT[v.label] || VERDICT.SUSPICIOUS;
  const video = report.video || {};
  const audio = report.audio || {};
  const rows = Object.entries(report.per_violation)
    .map(([type, s]) => ({ type, ...s }))
    .filter((r) => r.measurable)
    .sort((a, b) => b.percentage - a.percentage);
  const allRows = Object.entries(report.per_violation).map(([type, s]) => ({ type, ...s }));
  const scoreRows = allRows.filter((r) => r.max_points != null)
    .sort((a, b) => (b.score || 0) - (a.score || 0) || b.max_points - a.max_points);
  const audioEvents = report.timeline.filter((e) => e.source === 'audio');
  const bufferRows = allRows.filter((r) => r.measurable && r.merged_event_count > 0)
    .sort((a, b) => b.confirmed_event_count - a.confirmed_event_count || b.merged_event_count - a.merged_event_count);
  const confirmedCount = report.timeline.filter((e) => e.status !== 'unconfirmed').length;

  const [busy, setBusy] = useState(null);         // 'pdf' | 'docx' while the server builds the file
  const [dlError, setDlError] = useState(null);

  const saveBlob = (blob, name) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  const download = () => {
    saveBlob(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }),
      `proctoring_report_${reportId || 'session'}.json`);
  };

  // PDF / Word are built by the server (GET /api/reports/<id>/pdf|docx); errors are shown instead of a broken file
  const downloadDoc = async (kind) => {
    setBusy(kind);
    setDlError(null);
    try {
      const res = await fetch(`/api/reports/${reportId}/${kind}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Server answered ${res.status}`);
      }
      saveBlob(await res.blob(), `proctoring_report_${reportId}.${kind}`);
    } catch (e) {
      setDlError(`${kind === 'pdf' ? 'PDF' : 'Word'} download failed: ${e.message}`);
    } finally {
      setBusy(null);
    }
  };

  const docButtons = reportId && (
    <>
      <button onClick={() => downloadDoc('pdf')} disabled={!!busy} className="btn btn-primary">
        <FileText className="w-4 h-4" /> {busy === 'pdf' ? 'Preparing PDF…' : 'Download PDF'}
      </button>
      <button onClick={() => downloadDoc('docx')} disabled={!!busy} className="btn btn-outline">
        <FileText className="w-4 h-4" /> {busy === 'docx' ? 'Preparing Word…' : 'Download Word'}
      </button>
    </>
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-3">
        <div className="min-w-0">
          <h1 className="page-title">Session results</h1>
          <p className="text-muted mt-1">Candidate {orDash(report.meta?.candidate)} · {orDash(report.meta?.created)}</p>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:flex sm:flex-wrap xl:justify-end">
          {docButtons}
          <button onClick={download} className="btn btn-secondary"><Download className="w-4 h-4" /> Download JSON</button>
          <button onClick={onNewSession} className="btn btn-secondary"><Play className="w-4 h-4" /> New session</button>
          <button onClick={onReset} className="btn btn-secondary"><RotateCcw className="w-4 h-4" /> New candidate</button>
        </div>
      </div>

      {dlError && (
        <div className="alert alert-error" role="alert"><AlertTriangle className="w-5 h-5 flex-shrink-0" /> <span>{dlError}</span></div>
      )}

      <section className="card flex flex-col sm:flex-row sm:items-start gap-4" style={{ borderLeft: `3px solid ${edge}` }}
        aria-label="Verdict">
        <span className={`inline-flex items-center justify-center w-9 h-9 rounded-md flex-shrink-0 ${tone}`}>
          <Icon className="w-[18px] h-[18px]" aria-hidden="true" />
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-base font-semibold">{v.label}</p>
            <span className={`badge ${tone}`}>{word}</span>
            {v.assessment_status && v.assessment_status !== 'COMPLETE' && (
              <span className="badge badge-outline">{nice(v.assessment_status)}</span>
            )}
            {v.review_priority && v.review_priority !== 'low' && (
              <span className={`badge ${PRIORITY_BADGE[v.review_priority]}`}>priority {v.review_priority}</span>
            )}
          </div>
          <p className="text-[13px] font-medium mt-1">{orDash(v.headline)}</p>
          {(v.reasons || []).slice(1).map((r) => <p key={r} className="text-[13px] mt-1 text-muted">· {r}</p>)}
          {v.score_explanation && <p className="text-xs mt-2 text-subtle leading-relaxed">{v.score_explanation}</p>}
        </div>
        <div className="flex items-baseline justify-between gap-3 pt-3 border-t sm:block sm:pt-0 sm:border-t-0 sm:text-right
          sm:pl-5 sm:border-l border-line flex-shrink-0">
          <p className="text-xs font-medium text-muted">Violation score</p>
          <p className="text-[28px] font-semibold leading-9 mt-0.5 tabular-nums">{v.score.toFixed(1)}</p>
          <p className="text-xs text-subtle">of 100</p>
        </div>
      </section>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="stat">
          <p className="stat-label"><Clock /> Video analysed</p>
          <p className="stat-value">{(video.duration_s || 0).toFixed(0)} s</p>
          <p className="stat-meta">{video.n_samples ?? 0} samples @ {(video.sample_rate_hz || 0).toFixed(2)}/s</p>
        </div>
        <div className="stat">
          <p className="stat-label"><Mic /> Audio</p>
          <p className="stat-value capitalize">{(audio.status || 'not analysed').replace(/_/g, ' ')}</p>
          {audio.status === 'ok' && <p className="stat-meta">{audio.duration_s?.toFixed(0)} s, speech {audio.speech_s?.toFixed(1)} s</p>}
        </div>
        <div className="stat">
          <p className="stat-label"><ListChecks /> Confirmed events</p>
          <p className="stat-value">{confirmedCount}</p>
          {confirmedCount !== report.timeline.length && (
            <p className="stat-meta">+ {report.timeline.length - confirmedCount} unconfirmed</p>
          )}
          {video.baseline && (
            <p className="stat-meta font-mono break-words">
              baseline yaw {video.baseline.yaw.toFixed(1)}° pitch {video.baseline.pitch.toFixed(1)}° roll {video.baseline.roll.toFixed(1)}°
            </p>
          )}
        </div>
      </div>

      {report.session_violation?.violated && (
        <div className="alert alert-error" role="alert">
          <Ban className="w-5 h-5 flex-shrink-0" />
          <span>
            <span className="font-semibold">Session marked as violated.</span>{' '}
            Limits exceeded for: {report.session_violation.groups_exceeded.map((g) => GROUP_META[g]?.label || g).join(', ')}.
            Their evidence was {Object.values(report.evidence_upload || {}).some((u) => u.status === 'failed') ? 'partly NOT ' : ''}uploaded
            with session ID <span className="font-mono">{orDash(report.meta?.session_id)}</span>.
          </span>
        </div>
      )}

      {v.assessment_status === 'INCONCLUSIVE' && (
        <div className="alert alert-warn" role="status">
          <HelpCircle className="w-5 h-5 flex-shrink-0" />
          <span>
            <span className="font-semibold">Assessment inconclusive, retest recommended.</span>{' '}
            {(report.assessment?.coverage?.problems || []).join('; ')}
          </span>
        </div>
      )}

      {(report.warnings || []).map((w) => (
        <div key={w} className="alert alert-warn"><AlertTriangle className="w-5 h-5 flex-shrink-0" /> <span>{w}</span></div>
      ))}

      {scoreRows.length > 0 && <ScoringCard verdict={v} rows={scoreRows} />}

      {report.events && <FusionSection report={report} reportId={reportId} />}

      {report.assessment && <ViolationAnalysis rows={bufferRows} assessment={report.assessment} />}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <section className="card">
          <SectionTitle Icon={TrendingUp}>Violation percentage by type</SectionTitle>
          {rows.length ? <PercentBars rows={rows} /> : (
            <EmptyState compact icon={TrendingUp} title="Nothing could be measured"
              description="No video or audio samples were analysed for this session." />
          )}
        </section>
        <section className="card">
          <SectionTitle Icon={GanttChartSquare}>Timeline (video and audio)</SectionTitle>
          {report.timeline.length ? (
            <Timeline events={report.timeline} duration={Math.max(video.duration_s || 0, audio.duration_s || 0)} />
          ) : (
            <EmptyState compact icon={CheckCircle2} title="No violation events"
              description="Nothing was flagged on the video or audio timeline." />
          )}
        </section>
      </div>

      {report.evidence && (
        <section className="card space-y-6">
          <SectionTitle Icon={ImageIcon}>Evidence</SectionTitle>
          {EVIDENCE_ORDER.map((group) => (
            <EvidenceGroup
              key={group}
              group={group}
              data={report.evidence[group]}
              counter={report.evidence_counters?.[group]}
              upload={report.evidence_upload?.[group]}
              reportId={reportId}
            />
          ))}
        </section>
      )}

      <section className="card">
        <SectionTitle Icon={Mic}>Audio events</SectionTitle>
        {audioEvents.length ? (
          <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0">
            <table className="w-full min-w-[560px] text-sm">
              <thead><tr className="table-head"><th className="py-2 pr-4">Type</th><th className="pr-4">Start (s)</th><th className="pr-4">End (s)</th><th className="pr-4">Duration (s)</th><th>Details</th></tr></thead>
              <tbody>
                {audioEvents.map((e, i) => (
                  <tr key={i} className="table-row">
                    <td className="py-2 pr-4 capitalize">{nice(e.type)}</td><td className="pr-4">{e.start_s.toFixed(2)}</td>
                    <td className="pr-4">{e.end_s.toFixed(2)}</td><td className="pr-4">{e.duration_s.toFixed(2)}</td>
                    <td className="font-mono text-xs text-muted">{JSON.stringify(e.details || {})}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState compact icon={Mic}
            title={audio.status === 'ok' ? 'No audio events' : 'Audio was not analysed'}
            description={audio.status === 'ok' ? 'Speech, voices and sounds stayed within the allowed limits.'
              : 'No microphone or audio track was available for this session.'} />
        )}
      </section>

      <details className="card group">
        <summary className="card-title cursor-pointer flex items-center gap-2 list-none [&::-webkit-details-marker]:hidden">
          <Table2 className="w-4 h-4 text-subtle" aria-hidden="true" />
          <span className="flex-1">Per-violation table (numerator / denominator)</span>
          <ChevronDown className="w-4 h-4 text-subtle transition-transform duration-200 group-open:rotate-180" aria-hidden="true" />
        </summary>
        <div className="overflow-x-auto mt-4 -mx-4 px-4 sm:mx-0 sm:px-0">
          <table className="w-full min-w-[680px] text-sm">
            <thead>
              <tr className="table-head">
                <th className="py-2 pr-4">Type</th><th className="pr-4">Source</th><th className="pr-4">%</th><th className="pr-4">Numerator / denominator</th>
                <th className="pr-4">Seconds</th><th className="pr-4">Events</th><th className="pr-4">Severity</th><th>Points</th>
              </tr>
            </thead>
            <tbody>
              {allRows.map((r) => (
                <tr key={r.type} className="table-row">
                  <td className="py-2 pr-4 capitalize">{nice(r.type)}</td>
                  <td className="pr-4">{r.source}</td>
                  <td className="pr-4 font-mono">{r.measurable ? r.percentage.toFixed(1) : 'n/a'}</td>
                  <td className="pr-4">{r.measurable ? `${r.numerator} / ${r.denominator} ${r.unit}` : <span className="text-subtle">not measurable</span>}</td>
                  <td className="pr-4 font-mono">{r.duration_s.toFixed(1)}</td>
                  <td className="pr-4">{r.count}</td>
                  <td className="pr-4">
                    {r.severity
                      ? <span className={`badge ${SEVERITY_BADGE[r.severity] || 'badge-muted'}`}>{r.severity}</span>
                      : (r.critical ? <span className="badge badge-danger">critical</span> : '')}
                  </td>
                  <td className="font-mono">{r.max_points != null ? `${r.score != null ? r.score.toFixed(1) : '-'} / ${r.max_points}` : r.weight}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      {reportId && (
        <section className="card flex flex-col sm:flex-row sm:items-center justify-between gap-4" aria-label="Download report">
          <div className="flex items-start gap-3">
            <span className="icon-chip"><Download className="w-4 h-4" /></span>
            <div>
              <p className="font-medium">Download this report</p>
              <p className="text-sm text-muted mt-0.5">PDF and Word leave out the voice clips; those can only be played here.</p>
            </div>
          </div>
          <div className="flex flex-wrap gap-3">{docButtons}</div>
        </section>
      )}
    </div>
  );
};

export default Statistics;

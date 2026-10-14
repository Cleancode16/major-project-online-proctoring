import { useState } from 'react';
import {
  ShieldCheck, Flag, Ban, Download, RotateCcw, Play, AlertTriangle, TrendingUp, Clock, Mic, ListChecks,
  Image as ImageIcon, Table2, GanttChartSquare, Smartphone, Users, MoveHorizontal, UploadCloud, FileText, Gauge, ScanFace,
} from 'lucide-react';

// Results report (flow.txt PART II-D, D.3). Input: the report JSON built by server/aggregation/report.py.
// Series colours: CSS variables --series-video / --series-audio (index.css), light/dark steps of the
// reference palette slots 1-2. Chart text and grid lines follow the theme.
const SOURCE_VAR = { video: 'var(--series-video)', audio: 'var(--series-audio)' };
const VERDICT = {
  VALID: { cls: 'border-green-300 bg-green-50 text-green-900 dark:border-green-800 dark:bg-green-950/40 dark:text-green-100', Icon: ShieldCheck },
  SUSPICIOUS: { cls: 'border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-100', Icon: Flag },
  INVALID: { cls: 'border-red-300 bg-red-50 text-red-900 dark:border-red-800 dark:bg-red-950/40 dark:text-red-100', Icon: Ban },
};
const nice = (t) => t.toLowerCase().replace(/_/g, ' ');

const Legend = () => (
  <div className="flex gap-4 text-xs text-muted mb-2">
    {Object.entries(SOURCE_VAR).map(([k, c]) => (
      <span key={k} className="flex items-center gap-1.5">
        <span className="inline-block w-3 h-3 rounded-sm" style={{ background: c }} /> {k}
      </span>
    ))}
  </div>
);

const Tooltip = ({ tip }) =>
  tip ? (
    <div
      className="pointer-events-none fixed z-50 rounded-md px-2 py-1 text-xs shadow-lg whitespace-pre
        bg-slate-900 text-white ring-1 ring-slate-700 dark:bg-slate-100 dark:text-slate-900 dark:ring-slate-300"
      style={{ left: tip.x + 12, top: tip.y + 12 }}
    >
      {tip.text}
    </div>
  ) : null;

const PercentBars = ({ rows }) => {
  const [tip, setTip] = useState(null);
  const max = Math.max(10, ...rows.map((r) => r.percentage)) * 1.1;
  const rowH = 26;
  const labelW = 170;
  const width = 640;
  const plotW = width - labelW - 60;
  return (
    <div className="overflow-x-auto">
      <Legend />
      <svg viewBox={`0 0 ${width} ${rows.length * rowH + 24}`} className="w-full min-w-[480px]" role="img" aria-label="Violation percentage by type">
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
              <text x={labelW - 8} y={y + 12} fontSize="11" className="chart-text" textAnchor="end">{nice(r.type)}</text>
              <rect x={labelW} y={y + 2} width={w} height={14} rx={4} style={{ fill: SOURCE_VAR[r.source] }} />
              <text x={labelW + w + 6} y={y + 13} fontSize="11" className="chart-text-strong">{r.percentage.toFixed(1)}%</text>
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
  const types = [...new Set(events.map((e) => e.type))];
  const end = Math.max(duration || 0, ...events.map((e) => e.end_s), 1);
  const rowH = 24;
  const labelW = 170;
  const width = 640;
  const plotW = width - labelW - 20;
  const x = (t) => labelW + (t / end) * plotW;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * end);
  return (
    <div className="overflow-x-auto">
      <Legend />
      <svg viewBox={`0 0 ${width} ${types.length * rowH + 24}`} className="w-full min-w-[480px]" role="img" aria-label="Violation timeline">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={0} y2={types.length * rowH} className="chart-grid" />
            <text x={x(t)} y={types.length * rowH + 16} fontSize="10" className="chart-text" textAnchor="middle">{t.toFixed(0)} s</text>
          </g>
        ))}
        {types.map((t, i) => (
          <text key={t} x={labelW - 8} y={i * rowH + 16} fontSize="11" className="chart-text" textAnchor="end">{nice(t)}</text>
        ))}
        {events.map((e, k) => {
          const row = types.indexOf(e.type);
          const w = Math.max(3, x(Math.max(e.end_s, e.start_s + 0.2)) - x(e.start_s));
          return (
            <rect
              key={k}
              x={x(e.start_s)}
              y={row * rowH + 5}
              width={w}
              height={14}
              rx={3}
              style={{ fill: SOURCE_VAR[e.source] }}
              className="chart-gap"
              strokeWidth="1"
              onMouseMove={(ev) => setTip({ x: ev.clientX, y: ev.clientY, text:
                `${e.type} (${e.source})\n${e.start_s.toFixed(2)} - ${e.end_s.toFixed(2)} s (${e.duration_s.toFixed(2)} s)` +
                (e.details && Object.keys(e.details).length ? `\n${JSON.stringify(e.details)}` : '') })}
              onMouseLeave={() => setTip(null)}
            />
          );
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
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
    <span className={`badge text-xs ${style}`} title={upload.error || upload.target || upload.reason || ''}>
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
    <p className="text-sm">
      <span className="font-semibold">Overall score {verdict.score.toFixed(1)} / 100 ({verdict.label}).</span>{' '}
      <span className="text-muted">{verdict.score_explanation}</span>
    </p>
    <ul className="space-y-3">
      {rows.map((r) => (
        <li key={r.type} className="space-y-1">
          <div className="flex flex-wrap items-center gap-2 justify-between">
            <span className="font-medium capitalize inline-flex items-center gap-2">
              {nice(r.type)}
              <span className={`badge text-xs px-2 py-0.5 ${SEVERITY_BADGE[r.severity] || 'badge-muted'}`}>{r.severity}</span>
            </span>
            <span className="font-mono text-sm">{r.score != null ? r.score.toFixed(1) : '-'} / {r.max_points}</span>
          </div>
          <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden" aria-hidden="true">
            <div className="h-full rounded-full" style={{
              width: `${r.max_points ? (100 * (r.score || 0)) / r.max_points : 0}%`,
              background: r.critical ? 'rgb(220 38 38)' : SOURCE_VAR[r.source],
            }} />
          </div>
          <p className="text-xs text-muted">{r.score_explanation}</p>
        </li>
      ))}
    </ul>
  </section>
);

const EvidenceGroup = ({ group, data, counter, upload, reportId }) => {
  const { label, Icon } = GROUP_META[group];
  const items = data?.items || [];
  const fileUrl = (f) => `/api/reports/${reportId}/files/${f}`;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 justify-between">
        <h3 className="font-semibold inline-flex items-center gap-2"><Icon className="w-4 h-4" /> {label}</h3>
        <div className="flex items-center gap-2">
          {counter && (
            <span className={`badge text-xs ${counter.exceeded ? 'badge-danger' : 'badge-muted'}`}>
              {counterText(counter)}{counter.exceeded ? ' · exceeded' : ''}
            </span>
          )}
          <UploadBadge upload={upload} />
        </div>
      </div>
      {upload?.status === 'failed' && (
        <div className="alert alert-error text-xs"><AlertTriangle className="w-4 h-4 flex-shrink-0" /> Upload failed: {upload.error}</div>
      )}
      {items.length === 0 ? (
        <p className="text-sm text-subtle">No events.</p>
      ) : group === 'voice' ? (
        <ul className="space-y-2">
          {items.map((it) => (
            <li key={it.id} className="surface-muted rounded-lg p-3 flex flex-col md:flex-row md:items-center gap-3">
              <div className="md:w-64 flex-shrink-0">
                <p className="text-sm font-medium">{fmtT(it.start_s)} – {fmtT(it.end_s)}</p>
                <p className="text-xs text-muted">{itemDetails(group, it)}</p>
              </div>
              {it.file && reportId
                ? <audio controls preload="none" src={fileUrl(it.file)} className="w-full md:flex-1" />
                : <span className="text-xs text-subtle">{it.note || 'no clip'}</span>}
            </li>
          ))}
        </ul>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {items.map((it) => (
            <figure key={it.id} className="rounded-lg overflow-hidden ring-1 ring-slate-200 dark:ring-slate-800">
              {it.file && reportId ? (
                <a href={fileUrl(it.file)} target="_blank" rel="noreferrer">
                  <img src={fileUrl(it.file)} alt={`${label} at ${fmtT(it.start_s)}`} className="w-full" loading="lazy" />
                </a>
              ) : (
                <div className="aspect-video flex items-center justify-center text-xs text-subtle surface-muted">{it.note || 'no image'}</div>
              )}
              <figcaption className="text-xs px-2 py-1.5 surface-muted space-y-0.5">
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
  <h2 className="card-title flex items-center gap-2 mb-4">
    <Icon className="w-5 h-5 text-primary-600 dark:text-primary-400" /> {children}
  </h2>
);

const Statistics = ({ report, reportId, onNewSession, onReset }) => {
  const v = report.verdict;
  const { cls, Icon } = VERDICT[v.label] || VERDICT.SUSPICIOUS;
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
        <FileText className="w-4 h-4" /> {busy === 'pdf' ? 'Preparing PDF...' : 'Download PDF'}
      </button>
      <button onClick={() => downloadDoc('docx')} disabled={!!busy} className="btn btn-primary">
        <FileText className="w-4 h-4" /> {busy === 'docx' ? 'Preparing Word...' : 'Download Word'}
      </button>
    </>
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="page-title">Session results</h1>
          <p className="text-muted mt-1">Candidate {report.meta?.candidate} · {report.meta?.created}</p>
        </div>
        <div className="flex flex-wrap gap-3">
          {docButtons}
          <button onClick={download} className="btn btn-secondary"><Download className="w-4 h-4" /> Download JSON</button>
          <button onClick={onNewSession} className="btn btn-secondary"><Play className="w-4 h-4" /> New session</button>
          <button onClick={onReset} className="btn btn-secondary"><RotateCcw className="w-4 h-4" /> New candidate</button>
        </div>
      </div>

      {dlError && (
        <div className="alert alert-error" role="alert"><AlertTriangle className="w-5 h-5 flex-shrink-0" /> <span>{dlError}</span></div>
      )}

      <section className={`border-2 rounded-xl p-5 flex flex-col sm:flex-row sm:items-center gap-4 ${cls}`} aria-label="Verdict">
        <Icon className="w-12 h-12 flex-shrink-0" />
        <div className="flex-1">
          <p className="text-2xl font-bold tracking-tight">{v.label}</p>
          <p className="font-medium">{v.headline}</p>
          {v.reasons.slice(1).map((r) => <p key={r} className="text-sm mt-1 opacity-90">• {r}</p>)}
          {v.score_explanation && <p className="text-xs mt-2 opacity-80">{v.score_explanation}</p>}
        </div>
        <div className="sm:text-right">
          <p className="text-sm font-medium opacity-80">Violation score</p>
          <p className="text-4xl font-bold leading-none">{v.score.toFixed(1)}</p>
          <p className="text-xs opacity-80">of 100</p>
        </div>
      </section>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="stat stat-blue">
          <p className="stat-label inline-flex items-center gap-1"><Clock className="w-4 h-4" /> Video analysed</p>
          <p className="stat-value">{(video.duration_s || 0).toFixed(0)} s</p>
          <p className="text-xs opacity-80">{video.n_samples} samples @ {(video.sample_rate_hz || 0).toFixed(2)}/s</p>
        </div>
        <div className="stat stat-orange">
          <p className="stat-label inline-flex items-center gap-1"><Mic className="w-4 h-4" /> Audio</p>
          <p className="stat-value capitalize">{(audio.status || 'not analysed').replace(/_/g, ' ')}</p>
          {audio.status === 'ok' && <p className="text-xs opacity-80">{audio.duration_s?.toFixed(0)} s, speech {audio.speech_s?.toFixed(1)} s</p>}
        </div>
        <div className="stat stat-purple">
          <p className="stat-label inline-flex items-center gap-1"><ListChecks className="w-4 h-4" /> Events</p>
          <p className="stat-value">{report.timeline.length}</p>
          {video.baseline && (
            <p className="text-xs opacity-80 font-mono">
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
            with session ID <span className="font-mono">{report.meta?.session_id}</span>.
          </span>
        </div>
      )}

      {report.warnings.map((w) => (
        <div key={w} className="alert alert-warn"><AlertTriangle className="w-5 h-5 flex-shrink-0" /> <span>{w}</span></div>
      ))}

      {scoreRows.length > 0 && <ScoringCard verdict={v} rows={scoreRows} />}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <section className="card">
          <SectionTitle Icon={TrendingUp}>Violation percentage by type</SectionTitle>
          {rows.length ? <PercentBars rows={rows} /> : <p className="text-sm text-subtle">Nothing could be measured.</p>}
        </section>
        <section className="card">
          <SectionTitle Icon={GanttChartSquare}>Timeline (video and audio)</SectionTitle>
          {report.timeline.length ? (
            <Timeline events={report.timeline} duration={Math.max(video.duration_s || 0, audio.duration_s || 0)} />
          ) : (
            <p className="text-sm text-subtle">No violation events.</p>
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
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
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
          <p className="text-sm text-subtle">{audio.status === 'ok' ? 'No audio events.' : 'Audio was not analysed.'}</p>
        )}
      </section>

      <details className="card group">
        <summary className="card-title cursor-pointer flex items-center gap-2 list-none">
          <Table2 className="w-5 h-5 text-primary-600 dark:text-primary-400" /> Per-violation table (numerator / denominator)
        </summary>
        <div className="overflow-x-auto mt-4">
          <table className="w-full text-sm">
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
                      ? <span className={`badge text-xs px-2 py-0.5 ${SEVERITY_BADGE[r.severity] || 'badge-muted'}`}>{r.severity}</span>
                      : (r.critical ? <span className="badge badge-danger text-xs px-2 py-0.5">critical</span> : '')}
                  </td>
                  <td className="font-mono">{r.max_points != null ? `${r.score != null ? r.score.toFixed(1) : '-'} / ${r.max_points}` : r.weight}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      {reportId && (
        <section className="card flex flex-col sm:flex-row sm:items-center justify-between gap-3" aria-label="Download report">
          <p className="text-sm text-muted">Download this report. PDF and Word leave out the voice clips (they can only be played here).</p>
          <div className="flex flex-wrap gap-3">{docButtons}</div>
        </section>
      )}
    </div>
  );
};

export default Statistics;

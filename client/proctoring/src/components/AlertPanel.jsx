import { AlertTriangle, Eye, Users, Package, User, UserX, Mic, Volume2, EyeOff, Bell, ShieldAlert, ShieldCheck, MoveHorizontal } from 'lucide-react';
import EmptyState from './ui/EmptyState';

// Alert types come from the server's live status (server/webrtc_session.py ALERT_TEXT).
const TITLES = {
  identity: 'Identity Mismatch',
  pose: 'Head Pose Deviation',
  gaze: 'Looking Away',
  person: 'Multiple Persons Detected',
  object: 'Prohibited Object Detected',
  face: 'Face Not Visible',
  voice: 'Voice Mismatch',
  sound: 'Suspicious Sound',
  whisper: 'Whispering',
  spoof: 'Face Spoofing',
};
const CRITICAL = ['identity', 'person', 'object', 'voice', 'spoof'];

const ICONS = {
  identity: User, gaze: Eye, person: Users, object: Package, face: EyeOff, voice: UserX, whisper: Mic, sound: Volume2,
  spoof: ShieldAlert, pose: MoveHorizontal,
};

// tinted glass tone per alert family (index.css .tone-*)
const toneFor = (type) => {
  if (CRITICAL.includes(type)) return 'tone-danger';
  if (type === 'gaze') return 'tone-info';
  if (type === 'pose' || type === 'face') return 'tone-accent';
  return 'tone-warning';
};

const AlertPanel = ({ alerts }) => (
  <section className="card lg:sticky lg:top-28" aria-labelledby="alerts-title">
    <div className="flex items-center justify-between mb-4">
      <h2 id="alerts-title" className="card-title flex items-center gap-2.5">
        <span className="icon-chip w-8 h-8"><Bell className="w-4 h-4" /></span> Alerts
      </h2>
      {alerts.length > 0 && <span className="badge badge-danger tabular-nums">{alerts.length}</span>}
    </div>

    {alerts.length === 0 ? (
      <EmptyState
        icon={ShieldCheck}
        title="All clear"
        description="Alerts appear here as soon as a violation is confirmed during the session."
      />
    ) : (
      <ul className="space-y-2.5 max-h-[34rem] overflow-y-auto pr-1 -mr-1" aria-live="polite">
        {[...alerts].reverse().map((alert, index) => {
          const Icon = ICONS[alert.type] || AlertTriangle;
          const critical = CRITICAL.includes(alert.type);
          return (
            <li key={`${alert.type}-${alert.t}-${index}`} className={`rounded-2xl p-3.5 animate-slide-up ${toneFor(alert.type)}`}>
              <div className="flex gap-3">
                <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" aria-hidden="true" />
                <div className="flex-1 min-w-0">
                  <div className="flex justify-between items-start gap-2">
                    <span className="font-semibold text-sm">{TITLES[alert.type] || alert.type}</span>
                    <span className={`badge ${critical ? 'badge-danger' : 'badge-warning'} px-2 py-0.5 text-[0.7rem]`}>
                      {critical ? 'Critical' : 'Warning'}
                    </span>
                  </div>
                  <p className="text-sm mt-1 opacity-95">{alert.message || 'Violation detected'}</p>
                  <div className="flex justify-between gap-2 mt-2 text-xs opacity-80">
                    <span className="tabular-nums">t = {Number(alert.t ?? 0).toFixed(1)} s</span>
                    {alert.details && <span className="font-mono truncate">{alert.details}</span>}
                  </div>
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    )}
  </section>
);

export default AlertPanel;

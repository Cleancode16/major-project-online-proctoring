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

// Severity of each alert family: confirmed critical violations, warnings, and informational events (gaze).
// Rows stay neutral; only the icon and the small badge carry the colour, so a long list does not shout.
const SEVERITY = {
  critical: { label: 'Critical', badge: 'badge-danger', icon: 'text-critical' },
  warning: { label: 'Warning', badge: 'badge-warning', icon: 'text-warning' },
  info: { label: 'Info', badge: 'badge-info', icon: 'text-info' },
};
const severityOf = (type) => (CRITICAL.includes(type) ? 'critical' : type === 'gaze' ? 'info' : 'warning');

// FUS1 live lifecycle: provisional (debounce fired) -> confirmed (type rule met) -> ended (confirmed / not_confirmed).
// Alerts from older servers have no status and count as confirmed.
const isConfirmed = (a) => !a.status || a.status === 'confirmed';
const STAGE = {
  provisional: { label: 'Provisional', badge: 'badge-outline' },
  not_confirmed: { label: 'Not confirmed', badge: 'badge-muted' },
};

const AlertPanel = ({ alerts }) => {
  const nCritical = alerts.filter((a) => CRITICAL.includes(a.type) && isConfirmed(a)).length;
  return (
    <section className="card p-0 sm:p-0 lg:sticky lg:top-20" aria-labelledby="alerts-title">
      <div className="flex items-center justify-between px-4 h-11 border-b border-line">
        <h2 id="alerts-title" className="card-title flex items-center gap-2">
          <Bell className="w-4 h-4 text-subtle" aria-hidden="true" /> Alerts
        </h2>
        {alerts.length > 0 && (
          <span className="flex items-center gap-1.5">
            {nCritical > 0 && <span className="badge badge-danger tabular-nums">{nCritical} critical</span>}
            <span className="sr-only">{alerts.filter(isConfirmed).length} confirmed of</span>
            <span className="badge badge-muted tabular-nums">{alerts.length}</span>
          </span>
        )}
      </div>

      {alerts.length === 0 ? (
        <EmptyState
          icon={ShieldCheck}
          title="No alerts"
          description="Confirmed violations appear here during the session."
        />
      ) : (
        <ul className="max-h-[34rem] overflow-y-auto divide-y divide-[var(--border)]" aria-live="polite">
          {[...alerts].reverse().map((alert, index) => {
            const Icon = ICONS[alert.type] || AlertTriangle;
            const sev = SEVERITY[severityOf(alert.type)];
            const stage = isConfirmed(alert) ? null : STAGE[alert.status] || STAGE.provisional;
            return (
              <li key={alert.event_id || `${alert.type}-${alert.t}-${index}`}
                className={`px-4 py-3 flex gap-3 ${alert.status === 'not_confirmed' ? 'opacity-60' : ''}`}>
                <Icon className={`w-4 h-4 flex-shrink-0 mt-0.5 ${stage ? 'text-subtle' : sev.icon}`} aria-hidden="true" />
                <div className="flex-1 min-w-0">
                  <div className="flex justify-between items-center gap-2">
                    <span className="font-medium text-[13px] truncate">{TITLES[alert.type] || alert.type}</span>
                    <span className={`badge ${stage ? stage.badge : sev.badge}`}>{stage ? stage.label : sev.label}</span>
                  </div>
                  <p className="text-xs text-muted mt-0.5">{alert.message || 'Violation detected'}</p>
                  {alert.reason && <p className="text-2xs text-subtle mt-0.5">{alert.reason}</p>}
                  <div className="flex justify-between gap-2 mt-1 text-2xs text-subtle">
                    <span className="tabular-nums">t = {Number(alert.t ?? 0).toFixed(1)} s</span>
                    {alert.details && <span className="font-mono truncate">{alert.details}</span>}
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
};

export default AlertPanel;

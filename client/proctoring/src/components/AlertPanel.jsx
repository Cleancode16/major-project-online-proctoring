import { AlertTriangle, Eye, Users, Package, User, UserX, Mic, Volume2, EyeOff, Bell, ShieldAlert } from 'lucide-react';

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
  spoof: ShieldAlert,
};

const COLORS = {
  critical: 'border-red-200 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-200',
  orange: 'border-orange-200 bg-orange-50 text-orange-800 dark:border-orange-900 dark:bg-orange-950/40 dark:text-orange-200',
  blue: 'border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-900 dark:bg-blue-950/40 dark:text-blue-200',
  amber: 'border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200',
};

const AlertPanel = ({ alerts }) => {
  const colorFor = (type) => {
    if (CRITICAL.includes(type)) return COLORS.critical;
    if (type === 'pose' || type === 'face') return COLORS.orange;
    if (type === 'gaze') return COLORS.blue;
    return COLORS.amber;
  };

  return (
    <div className="card lg:sticky lg:top-24">
      <div className="flex items-center justify-between mb-4">
        <h2 className="card-title flex items-center gap-2"><Bell className="w-5 h-5" /> Alerts</h2>
        {alerts.length > 0 && <span className="badge badge-danger text-xs">{alerts.length}</span>}
      </div>

      {alerts.length === 0 ? (
        <div className="text-center py-10 text-subtle">
          <ShieldAlert className="w-12 h-12 mx-auto mb-2 opacity-40" />
          <p className="text-sm">No alerts yet. Monitoring in progress...</p>
        </div>
      ) : (
        <ul className="space-y-2 max-h-[32rem] overflow-y-auto pr-1" aria-live="polite">
          {[...alerts].reverse().map((alert, index) => {
            const Icon = ICONS[alert.type] || AlertTriangle;
            const critical = CRITICAL.includes(alert.type);
            return (
              <li key={`${alert.type}-${alert.t}-${index}`} className={`border rounded-lg p-3 ${colorFor(alert.type)}`}>
                <div className="flex gap-3">
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between items-start gap-2 mb-1">
                      <span className="font-semibold text-sm">{TITLES[alert.type] || alert.type}</span>
                      <span className={`badge text-xs px-2 py-0.5 text-white ${critical ? 'bg-red-600' : 'bg-amber-600'}`}>
                        {critical ? 'Critical' : 'Warning'}
                      </span>
                    </div>
                    <p className="text-sm">{alert.message}</p>
                    <div className="flex justify-between gap-2 mt-2 text-xs opacity-75">
                      <span>t = {Number(alert.t ?? 0).toFixed(1)} s</span>
                      {alert.details && <span className="font-mono truncate">{alert.details}</span>}
                    </div>
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};

export default AlertPanel;

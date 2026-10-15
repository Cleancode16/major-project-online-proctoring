import { Wifi, WifiOff, CheckCircle, Clock, Loader } from 'lucide-react';

// connectionState: RTCPeerConnection.connectionState ('new' | 'connecting' | 'connected' | 'disconnected' | 'failed' | 'closed')
const Pill = ({ tone, Icon, spin, pulse, children }) => (
  <span className={`badge ${tone} h-6 px-2 text-xs`}>
    <Icon className={`${spin ? 'animate-spin' : ''} ${pulse ? 'animate-pulse' : ''}`} aria-hidden="true" />
    {children}
  </span>
);

const StatusIndicator = ({ connectionState, calibrated, calibrationText }) => {
  const connected = connectionState === 'connected';
  const connecting = connectionState === 'connecting' || connectionState === 'new';
  return (
    <div className="flex flex-wrap items-center gap-2" role="status" aria-live="polite">
      {connected ? (
        <Pill tone="badge-success" Icon={Wifi}>Connected</Pill>
      ) : connecting ? (
        <Pill tone="badge-warning" Icon={Loader} spin>Connecting…</Pill>
      ) : (
        <Pill tone="badge-danger" Icon={WifiOff}>Not connected</Pill>
      )}
      {calibrated ? (
        <Pill tone="badge-success" Icon={CheckCircle}>Calibrated</Pill>
      ) : (
        <Pill tone="badge-warning" Icon={Clock} pulse>{calibrationText || 'Calibrating…'}</Pill>
      )}
    </div>
  );
};

export default StatusIndicator;

import { Wifi, WifiOff, CheckCircle, Clock, Loader } from 'lucide-react';

// connectionState: RTCPeerConnection.connectionState ('new' | 'connecting' | 'connected' | 'disconnected' | 'failed' | 'closed')
const StatusIndicator = ({ connectionState, calibrated, calibrationText }) => {
  const connected = connectionState === 'connected';
  const connecting = connectionState === 'connecting' || connectionState === 'new';
  const ok = 'text-green-700 dark:text-green-400';
  const wait = 'text-amber-700 dark:text-amber-400';
  const bad = 'text-red-700 dark:text-red-400';
  return (
    <div className="flex items-center gap-3 rounded-lg px-3 py-2 surface-muted ring-1 ring-slate-200 dark:ring-slate-800" role="status">
      <div className="flex items-center gap-2">
        {connected ? (
          <><Wifi className={`w-4 h-4 ${ok}`} /><span className={`text-sm font-medium ${ok}`}>Connected</span></>
        ) : connecting ? (
          <><Loader className={`w-4 h-4 animate-spin ${wait}`} /><span className={`text-sm font-medium ${wait}`}>Connecting...</span></>
        ) : (
          <><WifiOff className={`w-4 h-4 ${bad}`} /><span className={`text-sm font-medium ${bad}`}>Not connected</span></>
        )}
      </div>

      <div className="h-5 w-px bg-slate-300 dark:bg-slate-700"></div>

      <div className="flex items-center gap-2">
        {calibrated ? (
          <><CheckCircle className={`w-4 h-4 ${ok}`} /><span className={`text-sm font-medium ${ok}`}>Calibrated</span></>
        ) : (
          <><Clock className={`w-4 h-4 animate-pulse ${wait}`} /><span className={`text-sm font-medium ${wait}`}>{calibrationText || 'Calibrating...'}</span></>
        )}
      </div>
    </div>
  );
};

export default StatusIndicator;

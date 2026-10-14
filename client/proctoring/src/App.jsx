import { useEffect, useState } from 'react';
import { ShieldCheck, Sun, Moon, UserCheck, Video, BarChart3, Check, ChevronRight } from 'lucide-react';
import SetupScreen from './components/SetupScreen';
import ProctoringSession from './components/ProctoringSession';
import Statistics from './components/Statistics';
import './index.css';

// Steps (flow.txt PART II-D): enrollment -> test (live WebRTC or recorded video) -> results report
const STEPS = [
  { key: 'setup', label: 'Enrollment', Icon: UserCheck },
  { key: 'session', label: 'Test', Icon: Video },
  { key: 'results', label: 'Results', Icon: BarChart3 },
];

const initialTheme = () =>
  (typeof document !== 'undefined' && document.documentElement.classList.contains('dark') ? 'dark' : 'light');

const ThemeToggle = ({ theme, onToggle }) => {
  const dark = theme === 'dark';
  const label = dark ? 'Switch to light theme' : 'Switch to dark theme';
  return (
    <button onClick={onToggle} className="btn-icon" aria-label={label} title={label}>
      {dark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
    </button>
  );
};

const StepIndicator = ({ step }) => {
  const current = STEPS.findIndex((s) => s.key === step);
  return (
    <ol className="hidden md:flex items-center gap-1" aria-label="Progress">
      {STEPS.map(({ key, label, Icon }, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <li key={key} className="flex items-center gap-1">
            <span
              aria-current={active ? 'step' : undefined}
              className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                active
                  ? 'bg-primary-600 text-white'
                  : done
                  ? 'text-primary-700 dark:text-primary-300'
                  : 'text-slate-500 dark:text-slate-400'
              }`}
            >
              {done ? <Check className="w-4 h-4" /> : <Icon className="w-4 h-4" />}
              {label}
            </span>
            {i < STEPS.length - 1 && <ChevronRight className="w-4 h-4 text-slate-400 dark:text-slate-600" />}
          </li>
        );
      })}
    </ol>
  );
};

function App() {
  const [step, setStep] = useState('setup');
  const [candidate, setCandidate] = useState('');
  const [result, setResult] = useState(null); // { report, reportId }
  const [theme, setTheme] = useState(initialTheme);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    try {
      localStorage.setItem('theme', theme);
    } catch {
      /* storage blocked: the choice lasts for this page only */
    }
  }, [theme]);

  const handleEnrolled = (candidateId) => {
    setCandidate(candidateId);
    setStep('session');
  };

  const handleFinished = (report, reportId) => {
    setResult({ report, reportId });
    setStep('results');
  };

  const handleReset = () => {
    setResult(null);
    setCandidate('');
    setStep('setup');
  };

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-950/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2 min-w-0">
            <span className="inline-flex items-center justify-center w-9 h-9 rounded-lg bg-primary-600 text-white">
              <ShieldCheck className="w-5 h-5" />
            </span>
            <div className="min-w-0">
              <p className="font-semibold leading-tight truncate">Online Proctoring</p>
              <p className="text-xs text-muted truncate">AI exam monitoring · runs locally</p>
            </div>
          </div>
          <StepIndicator step={step} />
          <div className="flex items-center gap-3">
            {candidate && step !== 'setup' && (
              <span className="badge badge-muted hidden sm:inline-flex" title="Candidate">
                <UserCheck className="w-4 h-4" /> {candidate}
              </span>
            )}
            <ThemeToggle theme={theme} onToggle={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))} />
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {step === 'setup' && <SetupScreen onEnrolled={handleEnrolled} />}
        {step === 'session' && (
          <ProctoringSession candidate={candidate} onFinished={handleFinished} onReset={handleReset} />
        )}
        {step === 'results' && result && (
          <Statistics
            report={result.report}
            reportId={result.reportId}
            onNewSession={() => setStep('session')}
            onReset={handleReset}
          />
        )}
      </main>
    </div>
  );
}

export default App;

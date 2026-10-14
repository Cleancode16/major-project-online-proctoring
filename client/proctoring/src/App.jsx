import { useEffect, useState } from 'react';
import { ShieldCheck, Sun, Moon, UserCheck, Video, BarChart3, Check } from 'lucide-react';
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

// Desktop: pill steps. Mobile: "Step n of 3" with a thin progress bar (rendered under the header).
const StepIndicator = ({ step }) => {
  const current = STEPS.findIndex((s) => s.key === step);
  return (
    <ol className="hidden md:flex items-center gap-1 p-1 rounded-2xl" style={{ background: 'var(--fill)' }} aria-label="Progress">
      {STEPS.map(({ key, label, Icon }, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <li key={key}>
            <span
              aria-current={active ? 'step' : undefined}
              className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                active ? 'tab-active' : done ? 'text-primary-700 dark:text-primary-300' : 'text-subtle'
              }`}
            >
              <span className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-[0.7rem] ${
                active ? 'bg-gradient-to-br from-primary-600 to-accent-600 text-white'
                  : done ? 'bg-primary-500/15 text-primary-700 dark:text-primary-300' : 'surface-muted'
              }`}
              >
                {done ? <Check className="w-3 h-3" /> : <Icon className="w-3 h-3" />}
              </span>
              {label}
            </span>
          </li>
        );
      })}
    </ol>
  );
};

const MobileSteps = ({ step }) => {
  const current = STEPS.findIndex((s) => s.key === step);
  return (
    <div className="md:hidden max-w-7xl mx-auto px-4 pb-3" aria-label="Progress">
      <div className="flex items-center justify-between text-xs text-muted mb-1.5">
        <span className="font-medium" style={{ color: 'var(--text)' }}>{STEPS[current].label}</span>
        <span>Step {current + 1} of {STEPS.length}</span>
      </div>
      <div className="track h-1.5"><div className="track-fill" style={{ width: `${((current + 1) / STEPS.length) * 100}%` }} /></div>
    </div>
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

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [step]);

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
    <div className="min-h-screen flex flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 btn btn-secondary">
        Skip to content
      </a>
      <header className="sticky top-0 z-40 pt-3 px-3 sm:px-4">
        <div className="glass-strong rounded-3xl max-w-7xl mx-auto">
          <div className="px-4 sm:px-5 h-16 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <span className="inline-flex items-center justify-center w-10 h-10 rounded-2xl text-white shadow-glow
                bg-gradient-to-br from-primary-500 to-accent-600">
                <ShieldCheck className="w-5 h-5" />
              </span>
              <div className="min-w-0">
                <p className="font-semibold leading-tight truncate">Online Proctoring</p>
                <p className="text-xs text-muted truncate">AI exam monitoring · runs locally</p>
              </div>
            </div>
            <StepIndicator step={step} />
            <div className="flex items-center gap-2">
              {candidate && step !== 'setup' && (
                <span className="badge badge-muted hidden sm:inline-flex max-w-[12rem]" title="Candidate">
                  <UserCheck className="w-3.5 h-3.5 flex-shrink-0" /> <span className="truncate">{candidate}</span>
                </span>
              )}
              <ThemeToggle theme={theme} onToggle={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))} />
            </div>
          </div>
          <MobileSteps step={step} />
        </div>
      </header>

      <main id="main" className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 py-8">
        <div key={step} className="animate-slide-up">
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
        </div>
      </main>

      <footer className="max-w-7xl mx-auto w-full px-4 sm:px-6 pb-6 text-xs text-subtle flex flex-wrap gap-x-4 gap-y-1 justify-between">
        <span>Face, voice and scene analysis run on your local proctoring server.</span>
        <span>Evidence is shared only when a violation limit is exceeded.</span>
      </footer>
    </div>
  );
}

export default App;

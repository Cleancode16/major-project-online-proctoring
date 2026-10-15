import { useEffect, useState } from 'react';
import { ShieldCheck, Sun, Moon, UserCheck, Video, BarChart3, Check, ChevronRight, Cpu } from 'lucide-react';
import SetupScreen from './components/SetupScreen';
import ProctoringSession from './components/ProctoringSession';
import Statistics from './components/Statistics';
import './index.css';

// Steps (flow.txt PART II-D): enrollment -> test (live WebRTC or recorded video) -> results report.
// The steps are driven by the workflow (not free navigation), so the sidebar shows progress, not links.
const STEPS = [
  { key: 'setup', label: 'Enrollment', Icon: UserCheck },
  { key: 'session', label: 'Proctored test', Icon: Video },
  { key: 'results', label: 'Results', Icon: BarChart3 },
];

const initialTheme = () =>
  (typeof document !== 'undefined' && document.documentElement.classList.contains('dark') ? 'dark' : 'light');

const ThemeToggle = ({ theme, onToggle }) => {
  const dark = theme === 'dark';
  const label = dark ? 'Switch to light theme' : 'Switch to dark theme';
  return (
    <button onClick={onToggle} className="btn-icon glass-control" aria-label={label} title={label}>
      {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
    </button>
  );
};

const Brand = () => (
  <div className="flex items-center gap-2.5 min-w-0">
    <span className="inline-flex items-center justify-center w-7 h-7 rounded-md bg-accent text-accent-fg flex-shrink-0">
      <ShieldCheck className="w-4 h-4" aria-hidden="true" />
    </span>
    <div className="min-w-0 leading-tight">
      <p className="text-[13px] font-semibold truncate">Online Proctoring</p>
      <p className="text-2xs text-subtle truncate">Exam monitoring</p>
    </div>
  </div>
);

const StepNav = ({ step }) => {
  const current = STEPS.findIndex((s) => s.key === step);
  return (
    <nav aria-label="Workflow">
      <p className="px-2.5 mb-1.5 text-2xs font-medium uppercase tracking-wide text-subtle">Session</p>
      <ol className="space-y-0.5">
        {STEPS.map(({ key, label, Icon }, i) => {
          const done = i < current;
          return (
            <li key={key} className="nav-item" aria-current={i === current ? 'step' : undefined}>
              <Icon className="w-4 h-4 flex-shrink-0" aria-hidden="true" />
              <span className="flex-1 truncate">{label}</span>
              {done && <Check className="w-3.5 h-3.5 text-success" aria-label="completed" />}
            </li>
          );
        })}
      </ol>
    </nav>
  );
};

// avatar initials of the candidate ID (profile control; the app has no user accounts)
const CandidateChip = ({ candidate }) => (
  <span className="inline-flex items-center gap-2 h-8 pl-1 pr-2.5 rounded-md border border-line bg-surface max-w-[14rem]"
    title={`Candidate ${candidate}`}>
    <span className="inline-flex items-center justify-center w-6 h-6 rounded-sm bg-hover text-2xs font-semibold text-fg-muted">
      {candidate.slice(0, 2).toUpperCase()}
    </span>
    <span className="text-[13px] font-medium truncate">{candidate}</span>
  </span>
);

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
    window.scrollTo({ top: 0 });
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

  const current = STEPS.findIndex((s) => s.key === step);

  return (
    <div className="min-h-screen lg:flex">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 btn btn-secondary">
        Skip to content
      </a>

      {/* sidebar (desktop) */}
      <aside className="hidden lg:flex lg:flex-col w-60 flex-shrink-0 glass-sidebar h-screen sticky top-0 z-30">
        <div className="h-14 px-4 flex items-center border-b border-[var(--glass-divider)]"><Brand /></div>
        <div className="flex-1 overflow-y-auto px-2 py-4"><StepNav step={step} /></div>
        <div className="px-4 py-3 border-t border-[var(--glass-divider)] text-2xs text-subtle leading-4 space-y-1">
          <p className="flex items-center gap-1.5 text-fg-muted font-medium"><Cpu className="w-3.5 h-3.5" /> Runs on the local server</p>
          <p>Evidence is shared only when a violation limit is exceeded.</p>
        </div>
      </aside>

      <div className="flex-1 min-w-0 flex flex-col">
        {/* top bar */}
        <header className="glass-header sticky top-0 z-40 h-14">
          <div className="h-full px-4 sm:px-6 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="lg:hidden"><Brand /></div>
              <nav aria-label="Breadcrumb" className="hidden sm:flex items-center gap-1.5 text-[13px] min-w-0">
                <span className="text-subtle hidden lg:inline">Proctoring</span>
                <ChevronRight className="w-3.5 h-3.5 text-subtle hidden lg:inline" aria-hidden="true" />
                <span className="font-medium truncate" aria-current="page">{STEPS[current].label}</span>
                <span className="text-subtle lg:hidden">· Step {current + 1} of {STEPS.length}</span>
              </nav>
            </div>
            <div className="flex items-center gap-2">
              {candidate && step !== 'setup' && <span className="hidden sm:inline-flex"><CandidateChip candidate={candidate} /></span>}
              <ThemeToggle theme={theme} onToggle={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))} />
            </div>
          </div>
        </header>
        {/* mobile progress */}
        <div className="sm:hidden px-4 py-2 border-b border-line flex items-center justify-between text-xs">
          <span className="font-medium">{STEPS[current].label}</span>
          <span className="text-subtle">Step {current + 1} of {STEPS.length}</span>
        </div>

        <main id="main" className="flex-1 w-full max-w-6xl mx-auto px-4 sm:px-6 py-6">
          <div key={step} className="animate-fade-in">
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

        <footer className="lg:hidden px-4 sm:px-6 py-4 border-t border-line text-2xs text-subtle flex flex-wrap gap-x-4 gap-y-1 justify-between">
          <span>Face, voice and scene analysis run on your local proctoring server.</span>
          <span>Evidence is shared only when a violation limit is exceeded.</span>
        </footer>
      </div>
    </div>
  );
}

export default App;

// UI3 skeleton placeholders: shimmer blocks shaped like the content they replace (index.css .skeleton,
// shimmer disabled under prefers-reduced-motion). Purely visual, hidden from assistive technology.

export const Skeleton = ({ className = '' }) => <div className={`skeleton ${className}`} aria-hidden="true" />;

export const SkeletonText = ({ lines = 3, className = '' }) => (
  <div className={`space-y-2.5 ${className}`} aria-hidden="true">
    {Array.from({ length: lines }, (_, i) => (
      <Skeleton key={i} className={`h-3.5 ${i === lines - 1 ? 'w-2/3' : 'w-full'}`} />
    ))}
  </div>
);

const COLS = { 2: 'sm:grid-cols-2', 3: 'sm:grid-cols-3', 4: 'sm:grid-cols-4' };   // static names for Tailwind

export const SkeletonStatGrid = ({ count = 3 }) => (
  <div className={`grid grid-cols-1 ${COLS[count] || COLS[3]} gap-4`} aria-hidden="true">
    {Array.from({ length: count }, (_, i) => (
      <div key={i} className="rounded-2xl p-4 surface-muted space-y-3">
        <Skeleton className="h-3 w-24" />
        <Skeleton className="h-7 w-20" />
      </div>
    ))}
  </div>
);

// Same structure as the results page (Statistics.jsx): header, verdict banner, stat tiles, two chart cards.
export const ReportSkeleton = ({ label = 'Building the report...' }) => (
  <div className="space-y-6" role="status" aria-live="polite">
    <span className="sr-only">{label}</span>
    <div className="space-y-2">
      <Skeleton className="h-8 w-56" />
      <Skeleton className="h-4 w-72" />
    </div>
    <div className="card flex items-center gap-4">
      <Skeleton className="h-14 w-14 rounded-2xl" />
      <div className="flex-1 space-y-2">
        <Skeleton className="h-6 w-40" />
        <Skeleton className="h-4 w-3/4" />
      </div>
      <Skeleton className="h-12 w-20" />
    </div>
    <SkeletonStatGrid count={3} />
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
      {[0, 1].map((k) => (
        <div key={k} className="card space-y-4">
          <Skeleton className="h-5 w-48" />
          {[0, 1, 2, 3].map((i) => <Skeleton key={i} className={`h-4 ${i % 2 ? 'w-5/6' : 'w-full'}`} />)}
        </div>
      ))}
    </div>
  </div>
);

export default Skeleton;

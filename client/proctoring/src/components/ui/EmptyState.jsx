// UI3 empty state: icon chip, short title, one-line explanation and an optional action.
const EmptyState = ({ icon: Icon, title, description, action, compact = false, className = '' }) => (
  <div className={`flex flex-col items-center justify-center text-center ${compact ? 'py-6 gap-2' : 'py-10 gap-3'} ${className}`}>
    {Icon && (
      <span className={`icon-chip ${compact ? '' : 'w-12 h-12 rounded-2xl'}`}>
        <Icon className={compact ? 'w-4 h-4' : 'w-6 h-6'} aria-hidden="true" />
      </span>
    )}
    {title && <p className={`font-medium ${compact ? 'text-sm' : ''}`}>{title}</p>}
    {description && <p className="text-sm text-muted max-w-sm">{description}</p>}
    {action}
  </div>
);

export default EmptyState;

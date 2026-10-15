// UI4 empty state: neutral icon box, short title, one-line explanation and an optional action.
const EmptyState = ({ icon: Icon, title, description, action, compact = false, className = '' }) => (
  <div className={`flex flex-col items-center justify-center text-center ${compact ? 'py-6 gap-1.5' : 'py-10 gap-2'} ${className}`}>
    {Icon && (
      <span className={`icon-chip ${compact ? '' : 'w-9 h-9'}`}>
        <Icon className={compact ? 'w-4 h-4' : 'w-[18px] h-[18px]'} aria-hidden="true" />
      </span>
    )}
    {title && <p className="text-[13px] font-medium">{title}</p>}
    {description && <p className="text-xs text-muted max-w-xs">{description}</p>}
    {action}
  </div>
);

export default EmptyState;

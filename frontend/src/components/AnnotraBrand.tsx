/** Annotra logo + wordmark */
export default function AnnotraBrand({
  size = 'md',
  showTagline = false,
}: {
  size?: 'sm' | 'md' | 'lg';
  showTagline?: boolean;
}) {
  const img =
    size === 'lg' ? 'h-12 w-12' : size === 'sm' ? 'h-7 w-7' : 'h-9 w-9';
  const title =
    size === 'lg' ? 'text-xl' : size === 'sm' ? 'text-sm' : 'text-lg';

  return (
    <div className="flex items-center gap-3 min-w-0">
      <img
        src="/annotra-logo.png"
        alt="Annotra"
        className={`${img} object-contain shrink-0 rounded-lg`}
      />
      <div className="min-w-0">
        <p className={`font-semibold tracking-tight text-ocean-teal m-0 ${title}`}>Annotra</p>
        {showTagline ? (
          <p className="text-[10px] text-gray-500 uppercase tracking-wider truncate m-0 mt-0.5">
            Precision annotation for ML datasets
          </p>
        ) : null}
      </div>
    </div>
  );
}

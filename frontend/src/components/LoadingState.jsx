/**
 * LoadingState — displayed while the model is analyzing an image.
 */

export default function LoadingState() {
  return (
    <div id="loading-state" className="flex flex-col items-center gap-4 py-10">
      {/* Spinner */}
      <div className="relative w-12 h-12">
        <div className="absolute inset-0 rounded-full border-2 border-border" />
        <div className="absolute inset-0 rounded-full border-2 border-teal-500 border-t-transparent animate-spin" />
      </div>

      <div className="text-center">
        <p className="text-sm font-medium text-navy-700">
          Analyzing image…
        </p>
        <p className="text-xs text-navy-400 mt-1">
          Running Vision Transformer inference. This may take a moment.
        </p>
      </div>
    </div>
  );
}

/**
 * ResultDisplay — shows prediction results after analysis.
 *
 * Displays: original image, attention map, probability bars,
 * advisory text, and medical disclaimer.
 */

import { API_URL } from "../api/predict";

export default function ResultDisplay({ result, originalPreview }) {
  const {
    eye_laterality,
    probabilities,
    detections,
    advisory,
    heatmap_url,
    disclaimer,
  } = result;

  // Sort detections: detected first, then by probability descending
  const sortedDetections = [...detections].sort((a, b) => {
    if (a.detected !== b.detected) return b.detected ? 1 : -1;
    return b.probability - a.probability;
  });

  const detectedCount = detections.filter((d) => d.detected).length;

  return (
    <div className="w-full max-w-3xl mx-auto space-y-6">
      {/* Section title */}
      <h2
        id="results-heading"
        className="text-xl font-semibold text-navy-900 text-center"
      >
        Analysis Result
      </h2>

      {/* Eye Laterality Classification */}
      {eye_laterality && (
        <div
          id="eye-laterality-card"
          className="rounded-xl border border-teal-100 bg-teal-50/60 p-4 flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-teal-600/10 text-teal-600 flex items-center justify-center font-bold text-sm">
              {eye_laterality.code || "EYE"}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-teal-600">
                  Eye Classification
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-teal-100 text-teal-700 font-medium">
                  {Math.round((eye_laterality.confidence || 0.9) * 100)}% confidence
                </span>
              </div>
              <p className="text-sm font-semibold text-navy-900 mt-0.5">
                {eye_laterality.eye}
              </p>
              {eye_laterality.description && (
                <p className="text-xs text-navy-500 mt-0.5">
                  {eye_laterality.description}
                </p>
              )}
            </div>
          </div>
          <span className="text-xs font-mono text-navy-400 border border-border bg-white px-2.5 py-1 rounded-md">
            Disc: {eye_laterality.disc_side} side
          </span>
        </div>
      )}

      {/* Images: Original + Attention Map */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Original */}
        <div className="rounded-xl overflow-hidden border border-border bg-surface-card shadow-sm">
          <div className="px-4 py-2 border-b border-border-light bg-surface">
            <p className="text-xs font-medium text-navy-600">Original Image</p>
          </div>
          <img
            id="result-original-image"
            src={originalPreview}
            alt="Uploaded fundus image"
            className="w-full h-56 object-contain bg-navy-900/5"
          />
        </div>

        {/* Attention Map */}
        {heatmap_url && (
          <div className="rounded-xl overflow-hidden border border-border bg-surface-card shadow-sm">
            <div className="px-4 py-2 border-b border-border-light bg-surface">
              <p className="text-xs font-medium text-navy-600">
                AI Attention Map
              </p>
            </div>
            <img
              id="result-attention-map"
              src={`${API_URL}${heatmap_url}`}
              alt="ViT attention heatmap overlay"
              className="w-full h-56 object-contain bg-navy-900/5"
            />
          </div>
        )}
      </div>

      {/* Attention map explanation */}
      {heatmap_url && (
        <p className="text-xs text-navy-400 text-center italic px-4">
          Highlighted regions represent areas that received greater attention
          from the model. This visualization is for interpretability only and is
          not a clinical diagnostic image.
        </p>
      )}

      {/* Probability bars */}
      <div
        id="probability-bars"
        className="rounded-xl border border-border bg-surface-card shadow-sm overflow-hidden"
      >
        <div className="px-5 py-3 border-b border-border-light bg-surface">
          <p className="text-sm font-medium text-navy-700">
            Model Probability — All Classes
          </p>
          <p className="text-xs text-navy-400 mt-0.5">
            {detectedCount > 0
              ? `${detectedCount} pattern${detectedCount > 1 ? "s" : ""} detected above threshold`
              : "No patterns detected above threshold"}
          </p>
        </div>
        <div className="divide-y divide-border-light">
          {sortedDetections.map((d) => (
            <ProbabilityRow key={d.disease} detection={d} />
          ))}
        </div>
      </div>

      {/* Advisory */}
      {advisory && (
        <div
          id="advisory-section"
          className="rounded-xl border border-border bg-surface-card shadow-sm p-5"
        >
          <p className="text-sm font-medium text-navy-700 mb-2">Advisory</p>
          <p className="text-sm text-navy-600 leading-relaxed">{advisory}</p>
        </div>
      )}

      {/* Disclaimer */}
      {disclaimer && (
        <div
          id="disclaimer-section"
          className="rounded-xl border border-border bg-surface p-4"
        >
          <p className="text-xs text-navy-400 leading-relaxed text-center">
            {disclaimer}
          </p>
        </div>
      )}
    </div>
  );
}

/** Single probability bar row */
function ProbabilityRow({ detection }) {
  const { disease, probability, threshold, detected } = detection;
  const pct = Math.round(probability * 100);

  return (
    <div
      className={`px-5 py-3 flex items-center gap-4 ${
        detected ? "bg-detected-bg/40" : ""
      }`}
    >
      {/* Disease name */}
      <span
        className={`text-sm w-28 shrink-0 ${
          detected ? "font-semibold text-navy-900" : "text-navy-600"
        }`}
      >
        {disease}
        {detected && (
          <span className="ml-1.5 inline-block w-1.5 h-1.5 rounded-full bg-detected align-middle" />
        )}
      </span>

      {/* Bar */}
      <div className="flex-1 h-2 bg-border-light rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            detected ? "bg-detected" : "bg-teal-500"
          }`}
          style={{ width: `${Math.max(pct, 1)}%` }}
        />
      </div>

      {/* Percentage */}
      <span
        className={`text-sm w-12 text-right tabular-nums ${
          detected ? "font-semibold text-detected" : "text-navy-500"
        }`}
      >
        {pct}%
      </span>

      {/* Threshold indicator */}
      <span className="text-[10px] text-navy-400 w-16 text-right">
        thr: {Math.round(threshold * 100)}%
      </span>
    </div>
  );
}

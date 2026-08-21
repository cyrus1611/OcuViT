/**
 * OcuViT — Minimal Testing UI
 *
 * Single-page application:
 *   Upload fundus image → Analyze → Display real ViT predictions
 */

import { useState, useCallback } from "react";
import ImageUpload from "./components/ImageUpload";
import ResultDisplay from "./components/ResultDisplay";
import LoadingState from "./components/LoadingState";
import { predictImage } from "./api/predict";

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Called when user selects a valid image
  const handleImageSelected = useCallback((file) => {
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
  }, []);

  // Run analysis
  const handleAnalyze = useCallback(async () => {
    if (!selectedFile || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await predictImage(selectedFile);
      setResult(data);
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        "Unable to analyze the image. Please try again.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [selectedFile, loading]);

  // Reset for another analysis
  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError("");
  }, []);

  return (
    <div className="flex-1 flex flex-col">
      {/* ─── Header ─── */}
      <header className="border-b border-border bg-surface-card">
        <div className="max-w-3xl mx-auto px-5 py-5">
          <div className="flex items-center gap-3">
            {/* Logo mark */}
            <div className="w-9 h-9 rounded-lg bg-teal-600 flex items-center justify-center shrink-0">
              <svg
                className="w-5 h-5 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-semibold text-navy-900 leading-tight">
                OcuViT
              </h1>
              <p className="text-xs text-navy-400">
                Ophthalmic Disease Screening
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* ─── Main content ─── */}
      <main className="flex-1 px-5 py-8">
        <div className="max-w-3xl mx-auto space-y-6">
          {/* Show upload section when no result */}
          {!result && (
            <>
              <ImageUpload
                onImageSelected={handleImageSelected}
                disabled={loading}
              />

              {/* Loading state */}
              {loading && <LoadingState />}

              {/* Error message */}
              {error && (
                <div
                  id="analysis-error"
                  className="rounded-xl border border-detected/30 bg-detected-bg p-4 text-center"
                >
                  <p className="text-sm text-detected">{error}</p>
                </div>
              )}

              {/* Analyze button — visible only when image is selected and not loading */}
              {selectedFile && !loading && (
                <div className="flex justify-center">
                  <button
                    id="analyze-button"
                    onClick={handleAnalyze}
                    className="
                      px-8 py-3 rounded-xl
                      bg-teal-600 text-white font-medium text-sm
                      hover:bg-teal-500 active:bg-teal-700
                      transition-colors duration-150
                      shadow-sm hover:shadow
                      disabled:opacity-50 disabled:cursor-not-allowed
                    "
                    disabled={loading}
                  >
                    Analyze Image
                  </button>
                </div>
              )}
            </>
          )}

          {/* Show results */}
          {result && (
            <>
              <ResultDisplay result={result} originalPreview={preview} />

              <div className="flex justify-center pt-2">
                <button
                  id="reset-button"
                  onClick={handleReset}
                  className="
                    px-6 py-2.5 rounded-xl
                    border border-border text-navy-600 font-medium text-sm
                    hover:bg-surface-hover hover:border-navy-400
                    transition-colors duration-150
                  "
                >
                  Analyze Another Image
                </button>
              </div>
            </>
          )}
        </div>
      </main>

      {/* ─── Footer ─── */}
      <footer className="border-t border-border py-4 px-5">
        <p className="text-[11px] text-navy-400 text-center max-w-2xl mx-auto leading-relaxed">
          This AI tool is intended for educational and research screening
          purposes only. It does not provide a medical diagnosis and does not
          replace evaluation by a qualified healthcare professional.
        </p>
      </footer>
    </div>
  );
}

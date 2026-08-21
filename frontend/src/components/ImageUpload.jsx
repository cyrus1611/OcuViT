/**
 * ImageUpload — drag-and-drop or click-to-browse image upload component.
 *
 * Shows image preview after selection.
 * Validates file type (jpg/jpeg/png) and size (< 10 MB).
 */

import { useCallback, useRef, useState } from "react";

const ALLOWED_TYPES = ["image/jpeg", "image/png"];
const MAX_SIZE_MB = 10;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

export default function ImageUpload({ onImageSelected, disabled }) {
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const validateAndSet = useCallback(
    (file) => {
      setError("");

      if (!file) return;

      if (!ALLOWED_TYPES.includes(file.type)) {
        setError("Invalid file type. Please upload a JPG, JPEG, or PNG image.");
        return;
      }

      if (file.size > MAX_SIZE_BYTES) {
        setError(
          `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum size is ${MAX_SIZE_MB} MB.`
        );
        return;
      }

      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreview(url);
      setFileName(file.name);
      onImageSelected(file);
    },
    [onImageSelected]
  );

  const handleFileChange = (e) => {
    validateAndSet(e.target.files?.[0]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    validateAndSet(e.dataTransfer.files?.[0]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  return (
    <div className="w-full max-w-xl mx-auto">
      {/* Drop zone */}
      <div
        id="upload-dropzone"
        role="button"
        tabIndex={0}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) =>
          (e.key === "Enter" || e.key === " ") &&
          !disabled &&
          inputRef.current?.click()
        }
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          relative border-2 border-dashed rounded-xl p-8
          flex flex-col items-center justify-center gap-3
          transition-all duration-200 cursor-pointer
          ${dragOver
            ? "border-teal-500 bg-teal-50"
            : "border-border hover:border-teal-400 hover:bg-surface-hover"
          }
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
        `}
      >
        {/* Upload icon */}
        <svg
          className="w-10 h-10 text-navy-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M12 16V4m0 0l-4 4m4-4l4 4M4 20h16"
          />
        </svg>

        <div className="text-center">
          <p className="text-navy-700 font-medium text-sm">
            Upload fundus image
          </p>
          <p className="text-navy-400 text-xs mt-1">
            Drag & drop or click to browse · JPG, JPEG, PNG · Max {MAX_SIZE_MB}{" "}
            MB
          </p>
        </div>

        <input
          ref={inputRef}
          id="image-input"
          type="file"
          accept=".jpg,.jpeg,.png"
          className="hidden"
          onChange={handleFileChange}
          disabled={disabled}
        />
      </div>

      {/* Error message */}
      {error && (
        <p id="upload-error" className="text-detected text-sm mt-3 text-center">
          {error}
        </p>
      )}

      {/* Image preview */}
      {preview && (
        <div
          id="image-preview"
          className="mt-5 rounded-xl overflow-hidden border border-border bg-surface-card shadow-sm"
        >
          <img
            src={preview}
            alt={`Preview: ${fileName}`}
            className="w-full max-h-72 object-contain bg-navy-900/5"
          />
          <div className="px-4 py-2 border-t border-border-light">
            <p className="text-xs text-navy-500 truncate">{fileName}</p>
          </div>
        </div>
      )}
    </div>
  );
}

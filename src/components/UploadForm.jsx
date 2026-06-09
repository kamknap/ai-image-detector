import { useRef, useState } from "react";
import {
  MAX_FILE_SIZE_BYTES,
  MAX_FILE_SIZE_MB,
  ACCEPTED_TYPES,
} from "../config.js";

/**
 * Upload area with drag & drop and click-to-browse.
 * Validates type and size locally before handing the file to the parent.
 *
 * Props:
 *   - onFileSelected(file): called with a valid File
 *   - onValidationError(message): called when the chosen file is invalid
 *   - disabled: blocks interaction while a request is in flight
 */
export default function UploadForm({
  onFileSelected,
  onValidationError,
  disabled,
}) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  // Validate a file against the accepted types and size limit.
  function validate(file) {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      return "Unsupported format. Please upload a JPG, PNG or WebP image.";
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      return `File is too large. Maximum size is ${MAX_FILE_SIZE_MB} MB.`;
    }
    return null;
  }

  // Shared handler for both drop and file-input changes.
  function handleFile(file) {
    if (!file) return;
    const error = validate(file);
    if (error) {
      onValidationError(error);
      return;
    }
    onFileSelected(file);
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    handleFile(e.dataTransfer.files?.[0]);
  }

  function handleDragOver(e) {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  }

  function handleDragLeave(e) {
    e.preventDefault();
    setIsDragging(false);
  }

  function openFileDialog() {
    if (!disabled) inputRef.current?.click();
  }

  // Keyboard accessibility: trigger the file dialog with Enter / Space.
  function handleKeyDown(e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openFileDialog();
    }
  }

  return (
    <div
      className={`dropzone${isDragging ? " dropzone--active" : ""}${
        disabled ? " dropzone--disabled" : ""
      }`}
      onClick={openFileDialog}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={0}
      aria-label="Upload an image by dragging it here or clicking to browse"
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        className="dropzone__input"
        onChange={(e) => handleFile(e.target.files?.[0])}
        disabled={disabled}
      />

      <svg
        className="dropzone__icon"
        viewBox="0 0 24 24"
        width="40"
        height="40"
        aria-hidden="true"
      >
        <path
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 16V4m0 0L8 8m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"
        />
      </svg>

      <p className="dropzone__title">
        Drag &amp; drop an image, or <span>browse</span>
      </p>
      <p className="dropzone__hint">
        JPG, PNG or WebP · up to {MAX_FILE_SIZE_MB} MB
      </p>
    </div>
  );
}

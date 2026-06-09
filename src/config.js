/**
 * Central configuration.
 *
 * To point the app at a real backend, set VITE_API_URL in your environment
 * (.env file locally, or set before running `npm run build` when deploying).
 * This is the ONLY place that needs to change when the backend becomes available.
 */

// Base URL of the backend. Empty string => no real backend configured.
export const API_URL = import.meta.env.VITE_API_URL || "";

// Endpoint path that performs the analysis.
export const ANALYZE_PATH = "/analyze";

// Use the local mock when explicitly requested, or whenever no API_URL is set.
export const USE_MOCK =
  import.meta.env.VITE_USE_MOCK === "true" || API_URL === "";

// Upload constraints (kept in one place so the UI and validation stay in sync).
export const MAX_FILE_SIZE_MB = 10;
export const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

// Abort the request if the backend takes longer than this (milliseconds).
export const REQUEST_TIMEOUT_MS = 30_000;

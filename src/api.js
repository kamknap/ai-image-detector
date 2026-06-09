import {
  API_URL,
  ANALYZE_PATH,
  USE_MOCK,
  REQUEST_TIMEOUT_MS,
} from "./config.js";

/**
 * Sample response returned by the mock backend.
 * Matches the contract the real backend is expected to implement:
 *   { "result": "AI-generated" | "Real", "confidence": 0..1 }
 */
const MOCK_RESPONSE = {
  result: "AI-generated",
  confidence: 0.943,
};

/**
 * Mocked network call. Resolves after a short delay to mimic latency.
 * Swap this out automatically by setting VITE_API_URL (see config.js).
 */
function analyzeMock() {
  return new Promise((resolve) => {
    setTimeout(() => resolve(MOCK_RESPONSE), 800);
  });
}

/**
 * Real network call to POST {API_URL}/analyze with the image as
 * multipart/form-data under the field name "image".
 */
async function analyzeReal(file) {
  const formData = new FormData();
  formData.append("image", file);

  // Time-box the request so a hung backend doesn't freeze the UI.
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(`${API_URL}${ANALYZE_PATH}`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}`);
    }

    return await response.json();
  } catch (err) {
    // Normalise an aborted request into a clear, user-facing timeout error.
    if (err.name === "AbortError") {
      throw new Error("The request timed out. Please try again.");
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Public entry point used by the UI.
 * @param {File} file - the image selected by the user
 * @returns {Promise<{result: string, confidence: number}>}
 */
export function analyzeImage(file) {
  return USE_MOCK ? analyzeMock() : analyzeReal(file);
}

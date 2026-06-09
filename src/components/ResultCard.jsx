/**
 * Displays the analysis result: a label (Real / AI-generated) plus a
 * confidence score rendered both as text and as a horizontal bar.
 *
 * Props:
 *   - result: { result: string, confidence: number }  // confidence in 0..1
 */
export default function ResultCard({ result }) {
  // Treat anything that isn't clearly "real" as AI-generated for styling.
  const isAi = result.result.toLowerCase().includes("ai");

  // Convert the 0..1 confidence into a rounded percentage for display.
  const percent = Math.round(result.confidence * 1000) / 10; // 1 decimal place

  return (
    <div className={`result${isAi ? " result--ai" : " result--real"}`}>
      <div className="result__header">
        <span className="result__badge">{isAi ? "AI-generated" : "Real"}</span>
        <span className="result__label">{result.result}</span>
      </div>

      <div className="result__score">
        <div className="result__score-row">
          <span>Confidence</span>
          <strong>{percent}%</strong>
        </div>

        {/* Progress bar; width is driven by the confidence value. */}
        <div
          className="result__bar"
          role="progressbar"
          aria-valuenow={percent}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className="result__bar-fill"
            style={{ width: `${percent}%` }}
          />
        </div>
      </div>
    </div>
  );
}

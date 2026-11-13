import React, { useRef, useState } from "react";
import "./NumberPlateDetection.css";

const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8001";

export default function PpeDetection() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function selectFile(selected) {
    if (!selected) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(selected.type)) {
      setError("Please upload a JPG, PNG or WEBP image.");
      return;
    }

    if (selected.size > 10 * 1024 * 1024) {
      setError("Image must be smaller than 10 MB.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError("");
  }

  function reset() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  async function detectPPE() {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API}/api/ppe/detect`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "PPE detection failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Backend connection failed.");
    } finally {
      setLoading(false);
    }
  }

  const detections = result?.detections || [];
  const classCounts = result?.class_counts || {};
  const missingCounts = result?.missing_equipment_counts || {};

  const equipment = [
    ["Helmet", "helmet", "no_helmet"],
    ["Gloves", "glove", "no_glove"],
    ["Goggles", "goggles", "no_goggles"],
    ["Shoes", "shoes", "no_shoes"]
  ];

  return (
    <div className="anpr">
      <header className="anpr-hero">
        <div className="anpr-eyebrow">
          <span className="anpr-live-dot" />
          AISDS / INDUSTRIAL COMPUTER VISION
        </div>

        <h1>PPE Compliance Detection</h1>

        <p>
          AI-powered workplace protective equipment detection using YOLO.
          Identify visible PPE and potential missing equipment.
        </p>

        <div className="anpr-tags">
          <span>◈ YOLO Detection</span>
          <span>▣ PPE Analysis</span>
          <span>◉ Safety Monitoring</span>
        </div>
      </header>

      <div className="anpr-layout">
        <section className="anpr-panel">
          <div className="anpr-heading">
            <div className="anpr-heading-icon">⇧</div>
            <div>
              <h2>Upload Workplace Image</h2>
              <p>Choose an image containing workers or protective equipment.</p>
            </div>
          </div>

          <div
            className="anpr-dropzone"
            onClick={() => inputRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={e => {
              e.preventDefault();
              selectFile(e.dataTransfer.files?.[0]);
            }}
            role="button"
            tabIndex={0}
            onKeyDown={e => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
          >
            <input
              ref={inputRef}
              type="file"
              hidden
              accept=".jpg,.jpeg,.png,.webp"
              onChange={e => selectFile(e.target.files?.[0])}
            />

            {!preview ? (
              <>
                <div className="anpr-upload-symbol">↑</div>
                <h3>Drop your workplace image here</h3>
                <p>or click to browse files</p>
                <span>JPG, PNG, WEBP · Maximum 10 MB</span>
              </>
            ) : (
              <>
                <div className="anpr-upload-symbol">✓</div>
                <h3>Image selected successfully</h3>
                <p>Click to choose another image</p>
              </>
            )}
          </div>

          {preview && (
            <div className="anpr-preview-wrap">
              <div className="anpr-preview-heading">
                <span>IMAGE PREVIEW</span>
                <span>{file?.name}</span>
              </div>
              <img src={preview} alt="Workplace preview" />
              <p>{(file.size / 1048576).toFixed(2)} MB</p>
            </div>
          )}

          <div className="anpr-actions">
            <button
              className="anpr-detect-btn"
              onClick={detectPPE}
              disabled={!file || loading}
            >
              {loading ? "Processing image..." : "⌕ Detect PPE"}
            </button>

            <button
              className="anpr-reset-btn"
              onClick={reset}
              disabled={loading}
            >
              Reset
            </button>
          </div>

          {loading && (
            <div className="anpr-loading">
              <span className="anpr-spinner" />
              <span>YOLO PPE detection in progress...</span>
            </div>
          )}

          {error && <div className="anpr-error">⚠ {error}</div>}
        </section>

        <section className="anpr-panel">
          <div className="anpr-heading">
            <div className="anpr-heading-icon">⌗</div>
            <div>
              <h2>PPE Detection Results</h2>
              <p>Detected equipment and potential missing PPE.</p>
            </div>
          </div>

          {!result && !loading && (
            <div className="anpr-empty">
              <div>▤</div>
              <h3>No results yet</h3>
              <p>Upload a workplace image and start detection.</p>
            </div>
          )}

          {loading && !result && (
            <div className="anpr-empty">
              <span className="anpr-spinner" />
              <h3>Analysing workplace image...</h3>
              <p>The AI model is checking visible protective equipment.</p>
            </div>
          )}

          {result && (
            <>
              <div className="anpr-metrics">
                <div>
                  <span>TOTAL DETECTIONS</span>
                  <strong>{result.total_detections ?? detections.length}</strong>
                </div>
                <div>
                  <span>MISSING PPE DETECTIONS</span>
                  <strong>{Object.values(missingCounts).reduce(
                    (sum, value) => sum + Number(value || 0), 0
                  )}</strong>
                </div>
              </div>

              {result.annotated_image && (
                <div className="anpr-annotated">
                  <h3>AI Annotated Detection Image</h3>
                  <img
                    src={`data:image/jpeg;base64,${result.annotated_image}`}
                    alt="PPE detection result with bounding boxes"
                  />
                </div>
              )}

              <h3 className="anpr-result-heading">Equipment Summary</h3>

              {equipment.map(([label, detectedKey, missingKey]) => {
  const detected = Number(classCounts[detectedKey] || 0);
  const missing = Number(missingCounts[missingKey] || 0);

  const status = missing > 0 ? "MISSING" :
    detected > 0 ? "PRESENT" : "NOT DETECTED";

  return (
    <article className="anpr-result-card" key={label}>
      <div className="anpr-result-top">
        <span>{label.toUpperCase()}</span>
        <span className={
          status === "MISSING" ? "anpr-warning" :
          status === "PRESENT" ? "anpr-success" : ""
        }>
          {status === "MISSING" ? "● MISSING" :
           status === "PRESENT" ? "● PRESENT" :
           "● NOT DETECTED"}
        </span>
      </div>
      <div className="anpr-confidence">
        <div>
          <span>Present detections</span>
          <strong>{detected}</strong>
        </div>
        <div>
          <span>Missing detections</span>
          <strong>{missing}</strong>
        </div>
      </div>
    </article>
  );
})}

              <h3 className="anpr-result-heading">Individual Detections</h3>

              {detections.length === 0 ? (
                <div className="anpr-empty">
                  <h3>No PPE objects detected</h3>
                  <p>Try a clearer image with visible workers or equipment.</p>
                </div>
              ) : (
                detections.map((item, index) => (
                  <article className="anpr-result-card" key={index}>
                    <div className="anpr-result-top">
                      <span>DETECTION #{index + 1}</span>
                      <span className={
                        item.category === "missing"
                          ? "anpr-warning"
                          : "anpr-success"
                      }>
                        {item.category === "missing"
                          ? "● MISSING EQUIPMENT"
                          : "● EQUIPMENT DETECTED"}
                      </span>
                    </div>
                    <div className="anpr-confidence">
                      <div>
                        <span>Class</span>
                        <strong>{String(item.class_name || "Unknown").replaceAll("_", " ").toUpperCase()}</strong>
                      </div>
                      <div>
                        <span>Confidence</span>
                        <strong>{(Number(item.confidence || 0) * 100).toFixed(1)}%</strong>
                      </div>
                    </div>
                  </article>
                ))
              )}

              <p className="anpr-disclaimer">
                These are image-level object detections, not verified
                person-by-person compliance decisions. Review results visually
                before taking workplace safety action.
              </p>
            </>
          )}
        </section>
      </div>
    </div>
  );
}




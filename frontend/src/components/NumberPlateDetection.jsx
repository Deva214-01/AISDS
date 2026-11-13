import React, { useRef, useState } from "react";
import "./NumberPlateDetection.css";

const API = import.meta.env.VITE_API_BASE_URL || "https://aisds.onrender.com";

export default function NumberPlateDetection() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(-1);
  const inputRef = useRef(null);

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
    setCopied(-1);
  }

  function reset() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    setCopied(-1);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function detectPlates() {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API}/api/number-plate/detect`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Detection failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Backend connection failed.");
    } finally {
      setLoading(false);
    }
  }

  async function copyNumber(text, index) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(index);
    } catch {
      setError("Could not copy the number. Please copy it manually.");
    }
  }

  const detections = result?.detections || [];
  const readable = detections.filter(
    item => item.plate_text && item.ocr_status === "READ"
  ).length;

  return (
    <div className="anpr">
      <header className="anpr-hero">
        <div className="anpr-eyebrow">
          <span className="anpr-live-dot" />
          TRAFFICX AI / COMPUTER VISION
        </div>

        <h1>Number Plate Recognition</h1>

        <p>
          Intelligent vehicle identification with YOLO detection
          and EasyOCR text extraction.
        </p>

        <div className="anpr-tags">
          <span>â—ˆ YOLO Detection</span>
          <span>â–¤ EasyOCR</span>
          <span>â—‰ AI Recognition</span>
        </div>
      </header>

      <div className="anpr-layout">
        <section className="anpr-panel">
          <div className="anpr-heading">
            <div className="anpr-heading-icon">â‡§</div>
            <div>
              <h2>Upload Vehicle Image</h2>
              <p>Choose an image containing a visible number plate.</p>
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
                <div className="anpr-upload-symbol">â†‘</div>
                <h3>Drop your vehicle image here</h3>
                <p>or click to browse files</p>
                <span>JPG, PNG, WEBP Â· Maximum 10 MB</span>
              </>
            ) : (
              <>
                <div className="anpr-upload-symbol">âœ“</div>
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
              <img src={preview} alt="Vehicle preview" />
              <p>{(file.size / 1048576).toFixed(2)} MB</p>
            </div>
          )}

          <div className="anpr-actions">
            <button
              className="anpr-detect-btn"
              onClick={detectPlates}
              disabled={!file || loading}
            >
              {loading ? "Processing image..." : "âŒ•  Detect & Extract"}
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
              <span>YOLO detection and OCR recognition in progress...</span>
            </div>
          )}

          {error && <div className="anpr-error">âš  {error}</div>}
        </section>

        <section className="anpr-panel">
          <div className="anpr-heading">
            <div className="anpr-heading-icon">âŒ—</div>
            <div>
              <h2>Recognition Results</h2>
              <p>Vehicle registration details and confidence scores.</p>
            </div>
          </div>

          {!result && !loading && (
            <div className="anpr-empty">
              <div>â–¤</div>
              <h3>No results yet</h3>
              <p>Upload a vehicle image and start detection.</p>
            </div>
          )}

          {loading && !result && (
            <div className="anpr-empty">
              <span className="anpr-spinner" />
              <h3>Analysing vehicle...</h3>
              <p>First-time OCR loading may take longer.</p>
            </div>
          )}

          {result && (
            <>
              <div className="anpr-metrics">
                <div>
                  <span>PLATES DETECTED</span>
                  <strong>{result.total_plates ?? detections.length}</strong>
                </div>
                <div>
                  <span>NUMBERS READ</span>
                  <strong>{readable}</strong>
                </div>
              </div>

              {result.annotated_image && (
                <div className="anpr-annotated">
                  <h3>AI Detection Preview</h3>
                  <img
                    src={`data:image/jpeg;base64,${result.annotated_image}`}
                    alt="Annotated detection result"
                  />
                </div>
              )}

              <h3 className="anpr-result-heading">
                Extracted Registration Numbers
              </h3>

              {detections.length === 0 && (
                <div className="anpr-empty">
                  <h3>No number plates detected</h3>
                  <p>Try a clearer vehicle image.</p>
                </div>
              )}

              {detections.map((item, index) => (
                <article className="anpr-result-card" key={index}>
                  <div className="anpr-result-top">
                    <span>VEHICLE PLATE #{index + 1}</span>
                    <span className={
                      item.plate_text ? "anpr-success" : "anpr-warning"
                    }>
                      {item.plate_text ? "â— EXTRACTED" : "â— NOT READ"}
                    </span>
                  </div>

                  <div className="anpr-number-plate">
                    <div className="anpr-india">IND</div>
                    <strong>{item.plate_text || "UNREADABLE"}</strong>
                  </div>

                  {item.plate_text && (
                    <button
                      className="anpr-copy-btn"
                      onClick={() => copyNumber(item.plate_text, index)}
                    >
                      {copied === index ? "âœ“ Copied" : "â–£ Copy Number"}
                    </button>
                  )}

                  <div className="anpr-confidence">
                    <div>
                      <span>YOLO detection confidence</span>
                      <strong>
                        {(Number(item.confidence || 0) * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div className="anpr-progress">
                      <span style={{
                        width: `${Math.min(100, Math.max(0, Number(item.confidence || 0) * 100))}%`
                      }} />
                    </div>
                  </div>

                  <div className="anpr-confidence">
                    <div>
                      <span>OCR recognition confidence</span>
                      <strong>
                        {(Number(item.ocr_confidence || 0) * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div className="anpr-progress">
                      <span style={{
                        width: `${Math.min(100, Math.max(0, Number(item.ocr_confidence || 0) * 100))}%`
                      }} />
                    </div>
                  </div>

                  <p className="anpr-class">
                    Detection class: {item.class_name}
                  </p>
                </article>
              ))}

              <p className="anpr-disclaimer">
                OCR results should be visually verified before official use.
                Confidence scores do not guarantee registration accuracy.
              </p>
            </>
          )}
        </section>
      </div>
    </div>
  );
}



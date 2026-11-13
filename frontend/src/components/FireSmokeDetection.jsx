
import { useEffect, useMemo, useState } from "react";
import { Flame, CloudFog, Upload, ScanSearch, AlertCircle } from "lucide-react";
import "./FireSmokeDetection.css";

const API = import.meta.env.VITE_API_BASE_URL || "https://aisds.onrender.com";

export default function FireSmokeDetection() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [selectedType, setSelectedType] = useState("fire");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }

    const url = URL.createObjectURL(file);
    setPreview(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  const filteredDetections = useMemo(() => {
    if (!result) return [];
    return (result.detections || []).filter(
      (item) => item.class_name?.toLowerCase() === selectedType
    );
  }, [result, selectedType]);

  const selectedCount =
    selectedType === "fire"
      ? result?.fire_count ?? 0
      : result?.smoke_count ?? 0;

  const averageConfidence = filteredDetections.length
    ? (
        filteredDetections.reduce(
          (sum, item) => sum + item.confidence,
          0
        ) / filteredDetections.length
      ) * 100
    : 0;

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0];
    if (!selected) return;

    const validTypes = ["image/jpeg", "image/png", "image/webp"];

    if (!validTypes.includes(selected.type)) {
      setError("Please upload a JPG, JPEG, PNG or WEBP image.");
      return;
    }

    if (selected.size > 10 * 1024 * 1024) {
      setError("Image must be 10 MB or smaller.");
      return;
    }

    setFile(selected);
    setResult(null);
    setError("");
  };

  const runDetection = async (type) => {
    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    setSelectedType(type);
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API}/api/fire-smoke/detect`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || `Detection failed (${response.status})`
        );
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Unable to connect to the detection server.");
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setFile(null);
    setResult(null);
    setError("");
    setSelectedType("fire");

    const input = document.getElementById("fire-smoke-upload");
    if (input) input.value = "";
  };

  return (
    <section className="fire-smoke-page">
      <div className="fs-heading">
        <div>
          <span className="fs-eyebrow">ENVIRONMENTAL HAZARD MONITORING</span>
          <h2>Fire & Smoke Detection</h2>
          <p>Upload an image to identify potential fire and smoke hazards.</p>
        </div>
        <div className="fs-status">
          <span className="fs-status-dot" />
          AI MODEL READY
        </div>
      </div>

      <div className="fs-detection-buttons">
        <button
          className={`fs-detect-btn fire-btn ${selectedType === "fire" ? "selected" : ""}`}
          onClick={() => runDetection("fire")}
          disabled={loading || !file}
        >
          <Flame size={19} />
          Detect Fire
        </button>

        <button
          className={`fs-detect-btn smoke-btn ${selectedType === "smoke" ? "selected" : ""}`}
          onClick={() => runDetection("smoke")}
          disabled={loading || !file}
        >
          <CloudFog size={19} />
          Detect Smoke
        </button>
      </div>

      <div className="fs-workspace">
        <div className="fs-panel">
          <div className="fs-panel-heading">
            <Upload size={19} />
            <h3>Upload Image</h3>
          </div>

          <label className="fs-upload-area" htmlFor="fire-smoke-upload">
            <Upload size={30} />
            <strong>Choose an image</strong>
            <span>JPG, JPEG, PNG or WEBP Â· Maximum 10 MB</span>
            <input
              id="fire-smoke-upload"
              type="file"
              accept=".jpg,.jpeg,.png,.webp"
              onChange={handleFileChange}
            />
          </label>

          {file && (
            <div className="fs-file-name">
              <span>{file.name}</span>
              <button onClick={clearAll}>Remove</button>
            </div>
          )}

          {preview && (
            <div className="fs-image-preview">
              <h4>Original Image</h4>
              <img src={preview} alt="Uploaded original" />
            </div>
          )}
        </div>

        <div className="fs-panel">
          <div className="fs-panel-heading">
            <ScanSearch size={19} />
            <h3>Detection Results</h3>
          </div>

          {!result && !loading && (
            <div className="fs-empty-state">
              <ScanSearch size={38} />
              <strong>Waiting for image analysis</strong>
              <span>Upload an image and select Detect Fire or Detect Smoke.</span>
            </div>
          )}

          {loading && (
            <div className="fs-loading">
              <div className="fs-spinner" />
              <strong>Analysing image...</strong>
              <span>YOLO model is processing your image.</span>
            </div>
          )}

          {result && !loading && (
            <>
              <div className="fs-result-stats">
                <div className="fs-stat-card">
                  <span>{selectedType === "fire" ? "Fire Count" : "Smoke Count"}</span>
                  <strong>{selectedCount}</strong>
                </div>
                <div className="fs-stat-card">
                  <span>Average Confidence</span>
                  <strong>{averageConfidence.toFixed(1)}%</strong>
                </div>
                <div className="fs-stat-card">
                  <span>Total Detections</span>
                  <strong>{result.total_detections}</strong>
                </div>
              </div>

              <div className="fs-class-counts">
                <span>ðŸ”¥ Fire: <strong>{result.fire_count}</strong></span>
                <span>ðŸŒ«ï¸ Smoke: <strong>{result.smoke_count}</strong></span>
              </div>

              <div className="fs-image-preview">
                <h4>Annotated Image</h4>
                <img
                  src={`data:image/jpeg;base64,${result.annotated_image}`}
                  alt="AI annotated detection result"
                />
              </div>

              <div className="fs-detection-list">
                <h4>{selectedType === "fire" ? "Fire" : "Smoke"} Detections</h4>
                {filteredDetections.length === 0 ? (
                  <p>No {selectedType} detected in this image.</p>
                ) : (
                  filteredDetections.map((item, index) => (
                    <div className="fs-detection-row" key={index}>
                      <span>{item.class_name.toUpperCase()} #{index + 1}</span>
                      <strong>{(item.confidence * 100).toFixed(1)}%</strong>
                    </div>
                  ))
                )}
              </div>

              <p className="fs-model-info">Model: {result.model}</p>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="fs-error">
          <AlertCircle size={19} />
          {error}
        </div>
      )}

      <p className="fs-disclaimer">
        AI-generated detections are indicative only. Always verify hazards
        independently and follow appropriate emergency procedures.
      </p>
    </section>
  );
}


import {
  LayoutDashboard, Camera, ShieldAlert, CarFront, ScanLine,
  BrainCircuit, HardHat, Flame, UserRound, ChartNoAxesCombined,
  Settings, ChevronLeft, ChevronRight, CheckCircle2, ArrowRight,
  Car, Bike, Activity, ShieldCheck, Download
} from "lucide-react";
import PpeDetection from "./components/PpeDetection";
import NumberPlateDetection from './components/NumberPlateDetection';
import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import TrafficIntelligence from "./components/TrafficIntelligence";
import FireSmokeDetection from "./components/FireSmokeDetection";

import "./App.css";

const API = import.meta.env.VITE_API_BASE_URL || "https://aisds.onrender.com";

const isSupportedVideoFile = (file) =>
  Boolean(file?.name && /\.(mp4|avi|mov|mkv)$/i.test(file.name));


function TrafficHistoryChart({ history }) {
  const records = [...(history ?? [])]
    .filter((item) => Number.isFinite(Number(item.total_vehicles)))
    .slice(-7);

  if (records.length === 0) {
    return (
      <div className="history-chart-empty">
        <span></span>
        <strong>No historical data available</strong>
        <p>Upload and complete a traffic video analysis to populate this chart.</p>
      </div>
    );
  }

  const values = records.map((item) => Number(item.total_vehicles));
  const maxValue = Math.max(...values, 1);
  const points = values.map((value, index) => ({
    x: records.length === 1 ? 300 : 40 + (index * 520) / (records.length - 1),
    y: 165 - (value / maxValue) * 125,
    value,
    label: records[index].completed_at
      ? new Date(records[index].completed_at).toLocaleDateString()
      : `Analysis ${index + 1}`,
  }));

  const line = points.map((point) => `${point.x},${point.y}`).join(" ");

  return (
    <div className="history-chart">
      <div className="history-chart-heading">
        <div>
          <span>HISTORICAL ANALYSIS</span>
          <h3>Vehicles per analysis</h3>
        </div>
        <span className="history-chart-total">
          {records.length} RECORD{records.length === 1 ? "" : "S"}
        </span>
      </div>

      <div className="history-chart-scroll">
        <svg
          viewBox="0 0 600 205"
          role="img"
          aria-label="Total vehicles recorded in the latest saved analyses"
        >
          {[40, 80, 120, 160].map((y) => (
            <line
              key={y}
              x1="30"
              y1={y}
              x2="570"
              y2={y}
              stroke="currentColor"
              opacity="0.12"
              strokeDasharray="4 5"
            />
          ))}

          {points.length > 1 && (
            <polyline
              points={line}
              fill="none"
              stroke="#55d5ff"
              strokeWidth="3"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          )}

          {points.map((point, index) => (
            <g key={`${point.label}-${index}`}>
              <circle
                cx={point.x}
                cy={point.y}
                r="5"
                fill="#55d5ff"
                stroke="#101b2d"
                strokeWidth="2"
              />
              <text
                x={point.x}
                y={Math.max(point.y - 12, 15)}
                fill="currentColor"
                fontSize="11"
                textAnchor="middle"
              >
                {point.value}
              </text>
              <text
                x={point.x}
                y="190"
                fill="currentColor"
                opacity="0.7"
                fontSize="10"
                textAnchor="middle"
              >
                {points.length > 4
                  ? point.label.slice(0, 5)
                  : point.label}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <p className="history-chart-note">
        Each point represents one saved video analysis, not a live per-second count.
      </p>
    </div>
  );
}


function App() {

  const [traffic, setTraffic] = useState(null);

  const [activePage, setActivePage] = useState("Dashboard");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const [assistantQuestion, setAssistantQuestion] = useState("");

  const [assistantMessages, setAssistantMessages] = useState([]);

  const [connectionError, setConnectionError] = useState("");

  const [history, setHistory] = useState([]);

  const [uploadFile, setUploadFile] = useState(null);

  const [uploading, setUploading] = useState(false);

  const [uploadMessage, setUploadMessage] = useState("");

  const [videoVersion, setVideoVersion] = useState(0);

  const [liveCameraEnabled, setLiveCameraEnabled] = useState(false);

  const [liveCameraStats, setLiveCameraStats] = useState({

    status: "stopped",

    total_vehicles: 0,

    cars: 0,

    motorcycles: 0,

    buses: 0,

    trucks: 0,

    error: null,

  });

  const [liveCameraError, setLiveCameraError] = useState("");
  const [helmetAlerts, setHelmetAlerts] = useState([]);
  
const [trafficIntelligence, setTrafficIntelligence] = useState({
  status: "waiting",
  occupancy_percent: 0,
  congestion: "UNKNOWN",
  alert: null,
  last_updated: null,
  recent_events: [],
  vehicle_counts: {
    cars: 0,
    motorcycles: 0,
    buses: 0,
    trucks: 0,
  },
});

// Uploaded-video Accident Detection states
const [accidentFile, setAccidentFile] = useState(null);
const [accidentUploading, setAccidentUploading] = useState(false);
const [accidentJobId, setAccidentJobId] = useState("");
const [accidentJobStatus, setAccidentJobStatus] = useState("idle");
const [accidentJobMessage, setAccidentJobMessage] = useState("");
const [accidentReport, setAccidentReport] = useState(null);
const [accidentVideoUrl, setAccidentVideoUrl] = useState("");
const [accidentError, setAccidentError] = useState("");
const accidentVideoRef = useRef(null);
const uploadInputRef = useRef(null);
const accidentInputRef = useRef(null);



  const heavyVehicles = (traffic?.buses ?? 0) + (traffic?.trucks ?? 0);

  const speed = traffic?.average_speed > 0

    ? `${traffic.average_speed} project units`

    : "Not available";

  const fetchTraffic = useCallback(async () => {

    try {

      const response = await fetch(`${API}/api/traffic`);

      if (!response.ok) throw new Error(`Traffic API returned ${response.status}`);

      const data = await response.json();

      setTraffic(data);

      setConnectionError("");

      return data;

    } catch (error) {

      console.error("Traffic API error:", error);

      setConnectionError(error.message || "Cannot connect to backend.");

      return null;

    }

  }, []);

  const fetchHistory = useCallback(async () => {

    try {

      const response = await fetch(`${API}/api/traffic/history`);

      if (!response.ok) throw new Error(`History API returned ${response.status}`);

      const data = await response.json();

      if (Array.isArray(data)) setHistory(data);

    } catch (error) {

      console.error("History API error:", error);

    }

  }, []);

  useEffect(() => {

    let active = true;

    let previousStatus = null;

    const loadData = async () => {

      try {

        const response = await fetch(`${API}/api/traffic`);

        if (!response.ok) throw new Error(`Traffic API returned ${response.status}`);

        const data = await response.json();

        if (!active) return;

        if (data.status === "completed" && previousStatus !== "completed") {
          // Refresh the processed video once when a new analysis completes.
          setVideoVersion(Date.now());
        }

        previousStatus = data.status;
        setTraffic(data);

        setConnectionError("");

        if (data.status === "completed") {

          const historyResponse = await fetch(`${API}/api/traffic/history`);

          if (historyResponse.ok) {

            const records = await historyResponse.json();

            if (active && Array.isArray(records)) setHistory(records);

          }

        }

      } catch (error) {

        if (active) setConnectionError(error.message || "Cannot connect to backend.");

        console.error("Traffic API error:", error);

      }

    };

    loadData();

    const interval = setInterval(loadData, 3000);

    return () => {

      active = false;

      clearInterval(interval);

    };

  }, []);
  
useEffect(() => {
  let active = true;

  const loadTrafficIntelligence = async () => {
    try {
      const response = await fetch(
        `${API}/api/traffic/intelligence`
      );

      if (!response.ok) {
        throw new Error(
          `Traffic intelligence API returned ${response.status}`
        );
      }

      const data = await response.json();

      if (active) {
        setTrafficIntelligence(data);
      }
    } catch (error) {
      console.error("Traffic intelligence API error:", error);
    }
  };

  loadTrafficIntelligence();

  const interval = setInterval(loadTrafficIntelligence, 2000);

  return () => {
    active = false;
    clearInterval(interval);
  };
}, []);


  useEffect(() => {

    if (!liveCameraEnabled) return undefined;

    let active = true;

    const loadCameraStats = async () => {

      try {

        const response = await fetch(`${API}/api/camera/stats`);

        if (!response.ok) throw new Error(`Camera stats API returned ${response.status}`);

        const data = await response.json();

        if (!active) return;

        setLiveCameraStats(data);

        setLiveCameraError(data.error || "");

      } catch (error) {

        if (!active) return;

        setLiveCameraError(error.message || "Unable to read camera statistics.");

      }

    };

    loadCameraStats();

    const interval = setInterval(loadCameraStats, 1000);

    return () => {

      active = false;

      clearInterval(interval);

    };

  }, [liveCameraEnabled]);

    // NEW: Helmet alerts API polling
  useEffect(() => {
    if (!liveCameraEnabled) return undefined;

    let active = true;

    const loadHelmetAlerts = async () => {
      try {
        const response = await fetch(
          `${API}/api/camera/helmet-alerts`
        );

        if (!response.ok) {
          throw new Error(`Helmet API returned ${response.status}`);
        }

        const data = await response.json();

        if (active) {
          setHelmetAlerts(data.alerts || []);
        }
      } catch (error) {
        console.error("Helmet alerts API error:", error);
      }
    };

    loadHelmetAlerts();

    const interval = setInterval(loadHelmetAlerts, 2000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [liveCameraEnabled]);

  useEffect(() => {

    if (activePage !== "Cameras" && activePage !== "Traffic Intelligence" && liveCameraEnabled) {

      setLiveCameraEnabled(false);

    }

  }, [activePage, liveCameraEnabled]);
  
// Poll the uploaded accident-analysis job until it completes.
useEffect(() => {
  if (!accidentJobId || !["queued", "processing"].includes(accidentJobStatus)) {
    return undefined;
  }

  let active = true;
  const checkJob = async () => {
    try {
      const response = await fetch(`${API}/api/accident/jobs/${accidentJobId}`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || `Job request failed: ${response.status}`);
      }
      if (!active) return;

      setAccidentJobStatus(data.status || "processing");
      setAccidentJobMessage(data.message || "Analysing uploaded video...");

      if (data.status === "completed") {
        setAccidentReport(data.report || null);
        setAccidentVideoUrl(`${API}/api/accident/jobs/${accidentJobId}/video`);
        setAccidentJobMessage("Analysis completed. Review the possible collision alerts below.");
      } else if (data.status === "failed") {
        setAccidentError(data.error || data.message || "Accident analysis failed.");
      }
    } catch (error) {
      if (active) {
        setAccidentError(error.message || "Could not check analysis status.");
        setAccidentJobStatus("failed");
      }
    }
  };

  checkJob();
  const timer = window.setInterval(checkJob, 2000);
  return () => {
    active = false;
    window.clearInterval(timer);
  };
}, [accidentJobId, accidentJobStatus]);

const handleAccidentUpload = async () => {
  if (!accidentFile) {
    setAccidentError("Please select a traffic video first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", accidentFile);
  setAccidentUploading(true);
  setAccidentError("");
  setAccidentReport(null);
  setAccidentVideoUrl("");
  setAccidentJobId("");
  setAccidentJobStatus("uploading");
  setAccidentJobMessage("Uploading video...");

  try {
    const response = await fetch(`${API}/api/accident/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || data.error || `Upload failed: ${response.status}`);
    }

    setAccidentJobId(data.job_id);
    setAccidentJobStatus(data.status || "queued");
    setAccidentJobMessage(data.message || "Video uploaded. Analysis started.");
  } catch (error) {
    setAccidentJobStatus("failed");
    setAccidentError(error.message || "Could not upload video.");
    setAccidentJobMessage("Upload failed.");
  } finally {
    setAccidentUploading(false);
  }
};

const seekToAccidentAlert = (seconds) => {
  const video = accidentVideoRef.current;
  if (!video) return;
  video.currentTime = Math.max(0, Number(seconds) || 0);
  video.play().catch(() => {});
};

  const handleVideoUpload = async () => {

    if (!uploadFile) {

      setUploadMessage("Please select a video first.");

      return;

    }

    const formData = new FormData();

    formData.append("file", uploadFile);

    setUploading(true);

    setUploadMessage("Uploading video and starting AI analysis...");

    try {

      const response = await fetch(`${API}/api/traffic/upload`, {

        method: "POST",

        body: formData,

      });

      const data = await response.json();

      if (!response.ok) throw new Error(data.detail || data.error || "Video upload failed.");

      setUploadMessage(data.message || "Video uploaded. Analysis started.");

      await fetchTraffic();

      await fetchHistory();

    } catch (error) {

      console.error("Video upload error:", error);

      setUploadMessage(`Upload failed: ${error.message}`);

    } finally {

      setUploading(false);

    }

  };

  const handleDeleteHistory = async () => {
    if (history.length === 0) return;
    if (!window.confirm("Delete all saved analysis history? This cannot be undone.")) return;

    try {
      const response = await fetch(`${API}/api/traffic/history`, {
        method: "DELETE",
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.message || "Could not delete history.");
      }

      setHistory([]);
      setUploadMessage("Analysis history deleted successfully.");
    } catch (error) {
      console.error("Delete history error:", error);
      setUploadMessage(`Could not delete history: ${error.message}`);
    }
  };

  const handleAssistantAsk = (event) => {

    event.preventDefault();

    const originalQuestion = assistantQuestion.trim();

    const question = originalQuestion.toLowerCase();

    if (!question || !traffic) return;

    let answer;

    if (question.includes("total") || question.includes("vehicle")) {

      answer = `Current analysis: ${traffic.total_vehicles ?? 0} vehicles - ${traffic.cars ?? 0} cars, ${traffic.motorcycles ?? 0} motorcycles, ${traffic.buses ?? 0} buses and ${traffic.trucks ?? 0} trucks.`;

    } else if (question.includes("congestion") || question.includes("traffic density") || question.includes("traffic level")) {

      answer = `Congestion: ${traffic.congestion ?? "UNKNOWN"}. Traffic density: ${traffic.traffic_density ?? 0}%.`;

    } else if (question.includes("violation") || question.includes("crossing")) {

      answer = `${traffic.line_crossings ?? 0} line crossings reported. Line crossings alone do not prove a traffic-rule violation.`;

    } else if (question.includes("speed")) {

      answer = `Reported average speed: ${traffic.average_speed ?? 0} project units. This is not necessarily calibrated real-world speed.`;

    } else if (question.includes("history") || question.includes("analysis")) {

      answer = `Saved analysis records: ${history.length}. Open Analytics to see recent records.`;

    } else {

      answer = "Ask about vehicles, congestion, traffic density, line crossings, speed, or analysis history.";

    }

    setAssistantMessages((previous) => [...previous, { question: originalQuestion, answer }]);

    setAssistantQuestion("");

  };

  if (!traffic && connectionError) {

    return <div className="loading-screen"><div>

      <h1> AISDS</h1><p>Unable to connect to backend.</p>

      <small>Make sure FastAPI is running at https://aisds.onrender.com.</small><br />

      <button type="button" onClick={fetchTraffic}>Retry Connection</button>

    </div></div>;

  }

  if (!traffic) {

    return <div className="loading-screen"><div>

      <h1> AISDS</h1><p>Connecting to AI Safety Detection System...</p>

    </div></div>;

  }

  return (

    <div className="app">

      <aside className={`sidebar ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>

        <div className="logo"> <span>AISDS</span></div>
        <button type="button" className="sidebar-toggle" onClick={() => setSidebarCollapsed(!sidebarCollapsed)} aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"} title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}>{sidebarCollapsed
  ? <ChevronRight size={18} />
  : <ChevronLeft size={18} />}</button>

                <nav className="aisds-navigation">

          <button type="button"
            className={`nav-item ${activePage === "Dashboard" ? "active" : ""}`}
            onClick={() => setActivePage("Dashboard")}>
            <span className="nav-icon">
              <LayoutDashboard size={18} />
            </span>            <span>Dashboard</span>
          </button>

          <div className="nav-section-title">TRAFFIC SAFETY</div>

          {[
            ["Cameras", <Camera size={18}/>],
            ["Violations", <ShieldAlert size={18}/>],
            ["Crash Detection", <Activity size={18}/>],
            ["Number Plate Detection", <ScanLine size={18}/>],
            ["Traffic Intelligence", <BrainCircuit size={18}/>]
          ].map(([name, icon]) => (
            <button key={name} type="button"
              className={`nav-item ${activePage === name ? "active" : ""}`}
              onClick={() => setActivePage(name)}
              title={name}>
              <span className="nav-icon">{icon}</span>
              <span>{name}</span>
              {name === "Violations" && helmetAlerts.length > 0 && (
                <span className="nav-alert-count">{helmetAlerts.length}</span>
              )}
            </button>
          ))}

          <div className="nav-section-title">INDUSTRIAL SAFETY</div>

          {[
            ["PPE Detection", <HardHat size={18}/>],
            ["Fire & Smoke Detection", <Flame size={18}/>],
            ["Worker Safety", <ShieldCheck size={18}/>]
          ].map(([name, icon]) => (
            <button key={name} type="button"
              className={`nav-item ${activePage === name ? "active" : ""}`}
              onClick={() => setActivePage(name)}
              title={name}>
              <span className="nav-icon">{icon}</span>
              <span>{name}</span>
            </button>
          ))}

          <div className="nav-section-title">WORKSPACE</div>

          {[
            ["Analytics", <ChartNoAxesCombined size={18}/>],
            ["AI Assistant", <BrainCircuit size={18}/>]
          ].map(([name, icon]) => (
            <button key={name} type="button"
              className={`nav-item ${activePage === name ? "active" : ""}`}
              onClick={() => setActivePage(name)}
              title={name}>
              <span className="nav-icon">{icon}</span>
              <span>{name}</span>
            </button>
          ))}

        </nav>

<button
  type="button"
  className="settings"
  onClick={() => setUploadMessage("Settings are not configured yet.")}
>
  <Settings size={18} />
  <span>Settings</span>
</button>

      </aside>

      <main className="main">

        <header className="header">
          <div>
            <div className="header-eyebrow">TRAFFIC OPERATIONS CENTER</div>
            <h1>{activePage}</h1>
            <p>Monitor road activity, review important alerts and understand traffic conditions.</p>
          </div>
          <div className="header-right">
            <span className={`system-status ${liveCameraEnabled && liveCameraStats.status === "running" ? "is-live" : "is-idle"}`}>
              <span className="system-status-dot" />
              {liveCameraEnabled && liveCameraStats.status === "running" ? "CAMERA LIVE" : connectionError ? "BACKEND ISSUE" : "SYSTEM READY"}
            </span>
            <div className="profile">Admin</div>
          </div>
        </header>

        {connectionError && <div className="upload-message">Backend connection issue: {connectionError}</div>}

        {activePage === "Dashboard" && <>

          <section className={`priority-alert-panel ${helmetAlerts.length ? "has-alerts" : "no-alerts"}`} aria-live="polite">
            <div className="priority-alert-heading">
              <div className="priority-alert-symbol">{helmetAlerts.length ? "!" : <CheckCircle2 size={23}/>}</div>
              <div>
                <span className="priority-alert-kicker">PRIORITY SAFETY MONITOR</span>
                <h2>{helmetAlerts.length ? `${helmetAlerts.length} helmet alert${helmetAlerts.length === 1 ? "" : "s"} need review` : "No helmet alerts recorded"}</h2>
                <p>{helmetAlerts.length ? "Potential no-helmet detections from the live camera. Please verify before taking action." : "New potential no-helmet detections will appear here while live camera monitoring is enabled."}</p>
              </div>
<button
  type="button"
  className="priority-alert-action"
  onClick={() => setActivePage("Violations")}
>
  Review alerts <ArrowRight size={15} />
</button>            </div>
            {helmetAlerts.length > 0 && (
              <div className="priority-alert-latest">
                {helmetAlerts.slice(0, 2).map((alert, index) => (
                  <article className="priority-alert-item" key={alert.id || `${alert.timestamp}-${index}`}>
                    <span className="priority-alert-pulse" />
                    <div className="priority-alert-copy">
                      <strong>{alert.message || "Possible no-helmet violation detected"}</strong>
                      <small>{alert.timestamp ? new Date(alert.timestamp).toLocaleString() : "Time unavailable"}</small>
                    </div>
                    <span className="priority-alert-confidence">{Number.isFinite(Number(alert.confidence)) ? `${(Number(alert.confidence) * 100).toFixed(0)}% confidence` : "Review required"}</span>
                  </article>
                ))}
              </div>
            )}
          </section>

{/* AISDS - Command Center Hero */}





          <section className="upload-panel">

            <div className="upload-heading"><div><h3> Analyze Traffic Video</h3><p>Upload a video to detect and count vehicles.</p></div><span className="upload-tag">AI VIDEO ANALYSIS</span></div>

            <div className="upload-controls">

              <input
                ref={uploadInputRef}
                type="file"
                tabIndex={-1}
                aria-label="Choose traffic video"
                style={{
                  position: "absolute",
                  width: 1,
                  height: 1,
                  padding: 0,
                  margin: -1,
                  overflow: "hidden",
                  clip: "rect(0, 0, 0, 0)",
                  whiteSpace: "nowrap",
                  border: 0,
                }}
                onClick={(event) => { event.currentTarget.value = ""; }}
                onChange={(event) => {
                  const file = event.target.files?.[0] || null;
                  if (!file) return;
                  if (!isSupportedVideoFile(file)) {
                    setUploadFile(null);
                    setUploadMessage("Please choose an MP4, AVI, MOV, or MKV video.");
                    event.target.value = "";
                    return;
                  }
                  setUploadFile(file);
                  setUploadMessage("");
                }}
                disabled={uploading || traffic.worker_running}
              />

              <button
                type="button"
                onClick={() => uploadInputRef.current?.click()}
                disabled={uploading || traffic.worker_running}
              >
                Choose Video
              </button>

              <button type="button" onClick={handleVideoUpload} disabled={uploading || traffic.worker_running || !uploadFile}>

                {uploading ? "Uploading..." : "Upload & Analyze"}

              </button>

            </div>

            {uploadFile && <p className="selected-file">Selected: {uploadFile.name}</p>}

            {uploadMessage && <p className="upload-message">{uploadMessage}</p>}

          </section>

          <section className="stats dashboard-priority-stats">
            <div className="stat-card"><div className="stat-icon"><ShieldAlert size={22}/></div><div><p>Helmet Alerts</p><h2>{helmetAlerts.length}</h2><span>{helmetAlerts.length ? "Potential violations to review" : "No alerts recorded"}</span></div></div>
          </section>
          

          <section className="aisds-command-center">
            <div className="command-center-heading">
              <div>
                <span>UNIFIED SAFETY INTELLIGENCE</span>
                <h2>AISDS Command Center</h2>
                <p>Manage traffic and industrial safety from one place.</p>
              </div>
            </div>

            <div className="command-center-grid">
              <article className="command-center-card traffic-command">
                <div className="command-card-heading">
                  <span><HardHat size={24}/></span>
                  <div>
                    <h3>Traffic Safety</h3>
                    <small>Road monitoring and incident detection</small>
                  </div>
                  <span className="command-availability">5 MODULES</span>
                </div>
                <div className="command-links">
                  {[
                    ["Live Cameras", "Cameras"],
                    ["Helmet Violations", "Violations"],
                    ["Accident Detection", "Crash Detection"],
                    ["Number Plate Recognition", "Number Plate Detection"],
                    ["Traffic Intelligence", "Traffic Intelligence"]
                  ].map(([label, page]) => (
                    <button key={page} type="button" onClick={() => setActivePage(page)}>
                      {label}<ArrowRight size={15} />
                    </button>
                  ))}
                </div>
              </article>

              <article className="command-center-card industrial-command">
                <div className="command-card-heading">
                  <span><HardHat size={22} /></span>                  <div>
                    <h3>Industrial Safety</h3>
                    <small>Workplace protection and hazard monitoring</small>
                  </div>
                  <span className="command-planned">PLANNED</span>
                </div>
                <div className="command-links">
                  {[
                    ["PPE Compliance", "PPE Detection"],
                    ["Fire & Smoke Detection", "Fire & Smoke Detection"],
                    ["Worker Safety Monitoring", "Worker Safety"]
                  ].map(([label, page]) => (
                    <button key={page} type="button" onClick={() => setActivePage(page)}>
                      {label}<ArrowRight size={15} />
                    </button>
                  ))}
                </div>
              </article>
            </div>
          </section>
          <section className="dashboard-grid">

            <div className="panel camera-panel">

              <div className="panel-header"><div><h3> Processed Traffic Video</h3><p>AI vehicle detection output</p></div><span className="camera-live">{traffic.status === "completed" ? "VIDEO READY" : String(traffic.status ?? "idle").toUpperCase()}</span></div>

              <div className="camera">

                {traffic.status === "completed" ? <video key={videoVersion} src={`${API}/api/traffic/video?t=${videoVersion}`} controls muted playsInline preload="auto" style={{ width: "100%", height: "100%", objectFit: "contain", background: "#000" }} />

                  : <div className="camera-placeholder"><div className="camera-overlay">{traffic.status === "processing" ? "AI DETECTION PROCESSING..." : traffic.status === "error" ? "VIDEO PROCESSING FAILED" : "WAITING FOR TRAFFIC ANALYSIS"}</div></div>}

              </div>

              <div className="download-actions">

                <a href={`${API}/api/traffic/video/download`}><Download size={15}/> Download Processed Video</a>

                <a href={`${API}/api/traffic/report/download`}><Download size={15}/> Download Traffic Report</a>

              </div>

            </div>

            <div className="panel traffic-panel">

              <div className="panel-header"><div><h3>Traffic Status</h3><p>Current road condition</p></div></div>

              <div className="traffic-status"><div className="status-circle"><strong>{traffic.congestion ?? "UNKNOWN"}</strong><span>Congestion</span></div></div>

              <div className="traffic-info"><div><span>Average Speed</span><strong>{speed}</strong></div><div><span>Traffic Density</span><strong>{traffic.traffic_density ?? 0}%</strong></div></div>

            </div>

          </section>

          <section className="bottom-grid"><div className="panel analytics">

            <div className="panel-header"><div><h3> Traffic Analytics</h3><p>Current vehicle distribution</p></div></div>

            <div className="chart">


              <div
                className="bar bar1"
                style={{ height: `${Math.min((traffic.cars ?? 0) * 2, 100)}%` }}
              />

              <div className="bar bar2" style={{ height: `${Math.min((traffic.motorcycles ?? 0) * 5, 100)}%` }} />

              <div className="bar bar3" style={{ height: `${Math.min((traffic.buses ?? 0) * 5, 100)}%` }} />

              <div className="bar bar4" style={{ height: `${Math.min((traffic.trucks ?? 0) * 5, 100)}%` }} />

            </div>

            <div className="chart-labels"><span>Cars</span><span>Motorcycles</span><span>Buses</span><span>Trucks</span></div>

          </div></section>

          <section className="analysis-history">

            <div className="upload-heading">
              <div>
                <h2> Analysis History</h2>
                <p>Previously completed traffic analyses</p>
              </div>
              <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <span className="upload-tag">{history.length} RECORDS</span>
                <button type="button" onClick={handleDeleteHistory} disabled={history.length === 0}>
                  Delete History
                </button>
              </div>
            </div>

            {history.length === 0 ? <p>No previous analyses found.</p> : <div className="history-grid">

              {history.map((item, index) => <article className="history-card" key={item.completed_at || index}>

                <h3>Analysis #{history.length - index}</h3>

                <p><strong>Video:</strong> {item.video_filename || "Filename unavailable"}</p>

                <p><strong>Completed:</strong> {item.completed_at ? new Date(item.completed_at).toLocaleString() : "Unknown"}</p>

                <p><strong>Total Vehicles:</strong> {item.total_vehicles ?? 0}</p>

                <p>Cars: {item.cars ?? 0} · Motorcycles: {item.motorcycles ?? 0}</p>

                <p>Buses: {item.buses ?? 0} · Trucks: {item.trucks ?? 0}</p>

                <p><strong>Traffic Density:</strong> {item.traffic_density ?? 0}%</p>

                <p><strong>Congestion:</strong> {item.congestion ?? "UNKNOWN"}</p>

                <p><strong>Line Crossings:</strong> {item.line_crossings ?? 0}</p>

              </article>)}

            </div>}

          </section>

        </>}

        {activePage === "Cameras" && <section className="page-content">

          <h2> Camera Monitoring</h2>
          <section className="command-hero">
  <div className="hero-grid-glow" />

  <div className="hero-content">
    <div className="hero-eyebrow">
      <span className="hero-eyebrow-dot" />
      INTELLIGENT TRAFFIC OPERATIONS
    </div>

    <h2>
      See the city.
      <br />
      <span>Understand the flow.</span>
    </h2>

    <p className="hero-description">
      Real-time vehicle intelligence and congestion monitoring,
      powered by your traffic detection engine.
    </p>

    <div className="hero-status-row">
      <span className="hero-status-pill">
        <span className="hero-live-dot" />
        {trafficIntelligence.status === "running"
          ? "ENGINE ONLINE"
          : trafficIntelligence.status === "waiting"
            ? "AWAITING CAMERA"
            : String(trafficIntelligence.status ?? "UNKNOWN").toUpperCase()}
      </span>

      <span className="hero-updated">
        {trafficIntelligence.last_updated
          ? `Updated ${new Date(
              trafficIntelligence.last_updated
            ).toLocaleTimeString()}`
          : "Waiting for intelligence data"}
      </span>
    </div>

    <div className="hero-mini-stats">
      <div className="hero-mini-stat">
        <span>Current vehicles</span>
        <strong>
          {Object.values(
            trafficIntelligence.vehicle_counts ?? {}
          ).reduce((sum, value) => sum + (Number(value) || 0), 0)}
        </strong>
        <small>Latest detected frame</small>
      </div>

      <div className="hero-mini-stat">
        <span>Traffic events</span>
        <strong>
          {trafficIntelligence.recent_events?.length ?? 0}
        </strong>
        <small>Recent recorded transitions</small>
      </div>
    </div>
  </div>

  <div className="hero-gauge-area">
    <div className="hero-gauge-label">LIVE CONGESTION INDEX</div>

    <div
      className={`congestion-gauge ${
        String(trafficIntelligence.congestion ?? "UNKNOWN").toLowerCase()
      }`}
      style={{
        "--gauge-value": `${Math.min(
          100,
          Math.max(
            0,
            Number(trafficIntelligence.occupancy_percent) || 0
          )
        ) * 3.6}deg`,
      }}
      role="img"
      aria-label={`Estimated occupancy ${
        Number(trafficIntelligence.occupancy_percent ?? 0).toFixed(2)
      } percent, congestion ${
        trafficIntelligence.congestion ?? "UNKNOWN"
      }`}
    >
      <div className="congestion-gauge-inner">
        <span className="gauge-overline">OCCUPANCY</span>
        <strong>
          {Number(
            trafficIntelligence.occupancy_percent ?? 0
          ).toFixed(1)}
          <small>%</small>
        </strong>
        <span className="gauge-level">
          {trafficIntelligence.congestion ?? "UNKNOWN"}
        </span>
      </div>
    </div>

    <div className="gauge-scale">
      <span>0%</span>
      <span>50%</span>
      <span>100%</span>
    </div>

    <div className="hero-alert">
      <div className="hero-alert-icon"><Activity size={20} /></div>
      <div>
        <span>LAST TRAFFIC ALERT</span>
        <p>
          {trafficIntelligence.alert ||
            "No congestion transition reported yet."}
        </p>
      </div>
    </div>
  </div>
</section>
<section className="intelligence-full-width">
      <span className="camera-live">
        {String(trafficIntelligence.status ?? "UNKNOWN").toUpperCase()}
      </span>

    <div className="intelligence-content">
      <div className="intelligence-metrics">
        <div className="intelligence-metric">
          <p>Estimated Road Occupancy</p>
          <h2>
            {Number(trafficIntelligence.occupancy_percent ?? 0).toFixed(2)}%
          </h2>
        </div>
        <div className="intelligence-metric">
          <p>Congestion Level</p>
          <h2 className={`congestion-text ${String(trafficIntelligence.congestion ?? "UNKNOWN").toLowerCase()}`}>
            {trafficIntelligence.congestion ?? "UNKNOWN"}
          </h2>
        </div>
      </div>

      <div
        className="intelligence-occupancy-track"
        role="progressbar"
        aria-label="Estimated road occupancy"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.min(100, Math.max(0, Number(trafficIntelligence.occupancy_percent) || 0))}
      >
        <div
          className={`intelligence-occupancy-fill ${String(trafficIntelligence.congestion ?? "UNKNOWN").toLowerCase()}`}
          style={{
            width: `${Math.min(100, Math.max(0, Number(trafficIntelligence.occupancy_percent) || 0))}%`,
          }}
        />
      </div>

      <div className="intelligence-latest-alert">
        <div>
          <span className="intelligence-section-label">LATEST ALERT</span>
          <p>{trafficIntelligence.alert || "No new congestion transition reported."}</p>
        </div>
      </div>

      <div className="intelligence-events-section">
        <div className="intelligence-events-heading">
          <div>
            <h3>Recent Traffic Events</h3>
            <p>Latest congestion transitions recorded by the engine</p>
          </div>
          <span className="intelligence-event-count">
            {(trafficIntelligence.recent_events ?? []).length} EVENTS
          </span>
        </div>

        {(trafficIntelligence.recent_events ?? []).length === 0 ? (
          <div className="recent-events-empty">
            <span>—</span>
            <strong>No traffic events yet</strong>
            <p>Events will appear here when the congestion level changes.</p>
          </div>
        ) : (
          <div className="recent-events-list">
            {trafficIntelligence.recent_events.slice(0, 6).map((event, index) => {
              const level = String(event.congestion ?? "UNKNOWN").toLowerCase();
              return (
                <article
                  className="traffic-event"
                  key={`${event.timestamp ?? "event"}-${event.message ?? "message"}-${index}`}
                >
                  <div className="traffic-event-top">
                    <span className={`traffic-event-dot ${level}`} />
                    <strong>{event.congestion ?? "UNKNOWN"}</strong>
                    <time>
                      {event.timestamp
                        ? new Date(event.timestamp).toLocaleTimeString()
                        : "Time unavailable"}
                    </time>
                  </div>
                  <p>{event.message ?? "Congestion transition detected."}</p>
                </article>
              );
            })}
          </div>
        )}
      </div>

      <p className="intelligence-disclaimer">
        Occupancy is estimated from detected vehicle bounding boxes; it is not a calibrated measurement of actual road occupancy.
      </p>
    </div>
 </section>


          <p>Use your laptop's built-in webcam for live vehicle detection, or review the uploaded-video analysis.</p>

          <div className="camera-info-grid">

            <div className="camera-info-card">

              <h3> Laptop Live Camera</h3>

              <p>Status: <strong>{liveCameraEnabled ? (liveCameraStats.status || "starting") : "stopped"}</strong></p>

              <div style={{ width: "100%", minHeight: 240, background: "#10151f", borderRadius: 12, overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center", margin: "16px 0" }}>

                {liveCameraEnabled ? (

                  <img

                    src={`${API}/api/camera/live`}

                    alt="Live webcam with AI vehicle detections"

                    onError={() => setLiveCameraError("Could not load the camera stream. Check the backend terminal and camera permissions.")}

                    style={{ width: "100%", maxHeight: 520, objectFit: "contain" }}

                  />

                ) : (

                  <p style={{ color: "#d0d5dd", padding: 24, textAlign: "center" }}>Camera is stopped. Select Start Camera to begin live analysis.</p>

                )}

              </div>

              <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>

                {!liveCameraEnabled ? (

                  <button type="button" onClick={() => { setLiveCameraError(""); setLiveCameraEnabled(true); }}>

                    Start Camera

                  </button>

                ) : (

                  <button

                    type="button"

                    onClick={async () => {

                      try {

                        await fetch(`${API}/api/camera/stop`, {

                          method: "POST",

                        });

                      } catch (error) {

                        console.error("Camera stop error:", error);

                      } finally {

                        setLiveCameraEnabled(false);

                      }

                    }}

                  >

                    Stop Camera

                  </button>

                )}

              </div>

              {liveCameraError && <p className="upload-message">Camera issue: {liveCameraError}</p>}

              <p style={{ marginTop: 12, fontSize: 13 }}>

                Note: point the webcam toward a road with visible traffic. The current count shows vehicles detected in the latest frame, not unique cumulative vehicles.

              </p>

            </div>

            <div className="camera-info-card">

              <h3> Live Detection Statistics</h3>

              <p>Total vehicles in current frame: <strong>{liveCameraStats.total_vehicles ?? 0}</strong></p>

              <p> Cars: <strong>{liveCameraStats.cars ?? 0}</strong></p>

              <p> Motorcycles: <strong>{liveCameraStats.motorcycles ?? 0}</strong></p>

              <p> Buses: <strong>{liveCameraStats.buses ?? 0}</strong></p>

              <p> Trucks: <strong>{liveCameraStats.trucks ?? 0}</strong></p>

              <span className="camera-status">{liveCameraEnabled ? "Live analysis requested" : "Camera stopped"}</span>

            </div>

            <div className="camera-info-card">

              <h3>Sample / Uploaded Video</h3>

              <p>Status: {traffic.status ?? "Unknown"}</p>

              <p>Frames processed: {traffic.processed_frames ?? 0} / {traffic.total_frames ?? 0}</p>

              <p>Vehicles detected: {traffic.total_vehicles ?? 0}</p>

              <button type="button" onClick={() => setActivePage("Dashboard")}>Open Dashboard & Video</button>

            </div>

          </div>

        </section>}

        {activePage === "Violations" && <section className="page-content violations-page">
          <div className="page-intro-row">
            <div>
              <span className="section-kicker">SAFETY & COMPLIANCE</span>
              <h2>Violation Center</h2>
              <p>Review AI-generated safety alerts. Every alert requires human verification.</p>
            </div>
            <div className="violation-total-pill"><strong>{helmetAlerts.length}</strong><span>helmet alerts</span></div>
          </div>

          <div className="violation-grid">
            <div className="violation-card"><span>Potential No-Helmet Alerts</span><h3 className="violation-number-danger">{helmetAlerts.length}</h3><p>Alerts received from live camera analysis</p></div>
            <div className="violation-card"><span>Detected Line Crossings</span><h3>{traffic.line_crossings ?? 0}</h3><p>Events counted by the traffic engine</p></div>
            <div className="violation-card"><span>Camera Status</span><h3 className="violation-status-value">{liveCameraEnabled ? (liveCameraStats.status || "STARTING").toUpperCase() : "STOPPED"}</h3><p>Live camera monitoring state</p></div>
          </div>

          <section className="helmet-review-panel" aria-live="polite">
            <div className="helmet-review-heading">
              <div className="helmet-review-icon"><ShieldAlert size={20}/>Â </div>
              <div><span className="section-kicker">LIVE DETECTION FEED</span><h3>Helmet violation alerts</h3><p>Newest alerts appear first. Open each record and verify the detection.</p></div>
              <span className={`helmet-review-count ${helmetAlerts.length ? "active" : ""}`}>{helmetAlerts.length} {helmetAlerts.length === 1 ? "ALERT" : "ALERTS"}</span>
            </div>
            {helmetAlerts.length === 0 ? (
              <div className="helmet-empty-state"><span>CheckCircle2 size={20}</span><strong>No helmet alerts in this session</strong><p>Keep live camera monitoring enabled. Potential no-helmet detections will be listed here automatically.</p></div>
            ) : (
              <div className="helmet-alert-list">
                {helmetAlerts.map((alert, index) => (
                  <article className="helmet-alert-record" key={alert.id || `${alert.timestamp}-${index}`}>
                    <div className="helmet-alert-record-icon">!</div>
                    <div className="helmet-alert-record-main">
                      <div className="helmet-alert-record-title"><strong>{alert.message || "Possible no-helmet violation detected"}</strong><span>REVIEW REQUIRED</span></div>
                      <div className="helmet-alert-record-meta">
                        <span>Confidence: <strong>{Number.isFinite(Number(alert.confidence)) ? `${(Number(alert.confidence) * 100).toFixed(1)}%` : "Not available"}</strong></span>
                        <time>{alert.timestamp ? new Date(alert.timestamp).toLocaleString() : "Time unavailable"}</time>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>

          <div className="violation-note"><strong>Important:</strong> These are AI-generated potential violations, not confirmed offences. A human operator must review the camera footage and evidence before taking action. Line crossings alone do not establish a traffic-rule violation.</div>
        </section>}

        {activePage === "Analytics" && <section className="page-content">

          <h2> Traffic Analytics</h2><p>Current traffic analysis and saved results.</p>

          <div className="analytics-summary">

            <div className="analytics-card"><span>Total Vehicles</span><h3>{traffic.total_vehicles ?? 0}</h3></div>

            <div className="analytics-card"><span>Traffic Density</span><h3>{traffic.traffic_density ?? 0}%</h3></div>

            <div className="analytics-card"><span>Congestion</span><h3>{traffic.congestion ?? "UNKNOWN"}</h3></div>

            <div className="analytics-card"><span>Saved Analyses</span><h3>{history.length}</h3></div>

          </div>
          <TrafficHistoryChart history={history} />

          <h3 className="analytics-subheading">Vehicle Distribution</h3>

          {[

            { label: "Cars", value: traffic.cars ?? 0 },

            { label: "Motorcycles", value: traffic.motorcycles ?? 0 },

            { label: "Buses", value: traffic.buses ?? 0 },

            { label: "Trucks", value: traffic.trucks ?? 0 },

          ].map((vehicle) => (
            <div className="vehicle-bar-row" key={vehicle.label}>
              <span>{vehicle.label}</span>
              <div className="vehicle-bar-track">
                <div
                  className="vehicle-bar-fill"
                  style={{
                    width: `${Math.min(
                      (vehicle.value / Math.max(traffic.total_vehicles ?? 0, 1)) * 100,
                      100
                    )}%`,
                  }}
                />
              </div>
              <strong>{vehicle.value}</strong>
            </div>
          ))}

          <h3 className="analytics-subheading">Recent Analyses</h3>

          {history.length === 0 ? <p>No saved analyses available yet.</p> : <div className="analytics-history-list">

            {history.slice(0, 5).map((item, index) => <div className="analytics-history-row" key={item.completed_at || index}>

              <div><strong>{item.video_filename || `Analysis #${history.length - index}`}</strong><p>{item.completed_at ? new Date(item.completed_at).toLocaleString() : "Date unavailable"}</p></div>

              <strong>{item.total_vehicles ?? 0} vehicles</strong>

            </div>)}

          </div>}

        </section>}
        
{activePage === "Crash Detection" && (
  <section className="page-content accident-page">
    <div className="accident-heading">
      <div>
        <h2>Crash Detection</h2>
        <p>Upload a traffic video to detect suspicious vehicle interactions and review alerts directly on the annotated video.</p>
      </div>
    </div>

    <section className="accident-upload-panel">
      <div>
        <h3> Analyze a traffic video</h3>
        <p>Select an MP4, AVI, MOV, or MKV video. Processing may take a few minutes on CPU.</p>
      </div>
      <div className="accident-upload-controls">
        <input
          ref={accidentInputRef}
          type="file"
          tabIndex={-1}
          aria-label="Choose accident-analysis video"
          style={{
            position: "absolute",
            width: 1,
            height: 1,
            padding: 0,
            margin: -1,
            overflow: "hidden",
            clip: "rect(0, 0, 0, 0)",
            whiteSpace: "nowrap",
            border: 0,
          }}
          onClick={(event) => { event.currentTarget.value = ""; }}
          onChange={(event) => {
            const file = event.target.files?.[0] || null;
            if (!file) return;
            if (!isSupportedVideoFile(file)) {
              setAccidentFile(null);
              setAccidentError("Please choose an MP4, AVI, MOV, or MKV video.");
              event.target.value = "";
              return;
            }
            setAccidentFile(file);
            setAccidentError("");
          }}
          disabled={accidentUploading || ["queued", "processing"].includes(accidentJobStatus)}
        />
        <button
          type="button"
          onClick={() => accidentInputRef.current?.click()}
          disabled={accidentUploading || ["queued", "processing"].includes(accidentJobStatus)}
        >
          Choose Video
        </button>
        <button
          type="button"
          onClick={handleAccidentUpload}
          disabled={!accidentFile || accidentUploading || ["queued", "processing"].includes(accidentJobStatus)}
        >
          {accidentUploading ? "Uploading..." : "Analyze Video"}
        </button>
      </div>
      {accidentFile && <p className="accident-selected-file">Selected: {accidentFile.name}</p>}
      {accidentJobMessage && <p className="accident-job-message">{accidentJobMessage}</p>}
      {accidentJobId && <p className="accident-job-id">Job ID: {accidentJobId}</p>}
      {(["queued", "processing", "uploading"].includes(accidentJobStatus)) && (
        <div className="accident-processing"><span className="accident-spinner" /> AI is analysing the video. Please keep this page open.</div>
      )}
      {accidentError && <p className="accident-error">{accidentError}</p>}
    </section>

    {accidentReport && (
      <>
        <section className="accident-video-panel">
          <div className="accident-video-heading">
            <div>
              <h3>Annotated video</h3>
              <p>Red boxes and the warning banner are embedded in the processed video.</p>
            </div>
            <span className={`accident-status ${String(accidentReport.accident_status || "review_required").toLowerCase()}`}>
              {String(accidentReport.accident_status || "REVIEW_REQUIRED").replaceAll("_", " ")}
            </span>
          </div>
          {accidentVideoUrl ? (
            <video
              ref={accidentVideoRef}
              className="accident-video-player"
              src={accidentVideoUrl}
              controls
              playsInline
              preload="metadata"
            />
          ) : <p className="accident-error">Annotated video is not available.</p>}
        </section>

        <section className="accident-alert-panel">
          <div className="accident-alert-heading">
            <div>
              <h3>Possible collision alerts</h3>
              <p>Click an alert to jump to its timestamp in the video.</p>
            </div>
            <strong>{accidentReport.alert_count ?? accidentReport.collision_alerts?.length ?? 0} alerts</strong>
          </div>
          {accidentReport.collision_alerts?.length ? (
            <div className="accident-alert-list">
              {accidentReport.collision_alerts.map((alert, index) => (
                <button
                  type="button"
                  className="accident-alert-row"
                  key={`${alert.timestamp}-${alert.vehicle_a}-${alert.vehicle_b}-${index}`}
                  onClick={() => seekToAccidentAlert(alert.timestamp)}
                >
                  <span className="accident-alert-icon">!</span>
                  <span className="accident-alert-main">
                    <strong>{alert.vehicle_a_type} #{alert.vehicle_a} + {alert.vehicle_b_type} #{alert.vehicle_b}</strong>
                    <small>{alert.evidence || "Repeated bounding-box overlap"}</small>
                  </span>
<span className="accident-alert-time">
  {Number(alert.timestamp).toFixed(2)}s
</span>                </button>
              ))}
            </div>
          ) : (
            <p className="accident-no-alerts">No repeated-overlap alerts were detected in this video. This does not guarantee that no accident occurred.</p>
          )}
        </section>

        <div className="accident-disclaimer">
          <strong>Important: Human review required</strong>
          <p>These are experimental visual alerts based on repeated bounding-box overlap. They can produce false positives in dense traffic and do not confirm an accident. Always review the video before drawing conclusions.</p>
        </div>
      </>
    )}
  </section>
)}
{activePage === "Traffic Intelligence" && (
  <TrafficIntelligence
    cameraEnabled={liveCameraEnabled}
    cameraStats={liveCameraStats}
    cameraError={liveCameraError}
    helmetAlerts={helmetAlerts}
    onStartCamera={() => {
      setLiveCameraError("");
      setLiveCameraEnabled(true);
    }}
    onStopCamera={async () => {
      try {
        await fetch(`${API}/api/camera/stop`, {
          method: "POST",
        });
      } catch (error) {
        console.error("Camera stop error:", error);
      } finally {
        setLiveCameraEnabled(false);
      }
    }}
  />
)}


        {activePage === "Number Plate Detection" && <NumberPlateDetection />}


        {activePage === "Traffic Safety" && (
          <section className="safety-page">
            <div className="safety-page-heading">
              <span>TRAFFIC SAFETY INTELLIGENCE</span>
              <h2>Road Safety Command Center</h2>
              <p>Monitor traffic risks, road incidents and vehicle safety.</p>
            </div>
            <div className="safety-module-grid">
              {[
                ["Live Camera Monitoring", "Monitor connected traffic cameras.", "Available", "Cameras"],
                ["Helmet Violation Detection", "Review potential no-helmet alerts.", "Available", "Violations"],
                ["Accident Detection", "Analyze uploaded videos for possible collisions.", "Available", "Crash Detection"],
                ["Number Plate Recognition", "Detect and read vehicle registration plates.", "Available", "Number Plate Detection"],
                ["Traffic Intelligence", "Review road occupancy and congestion.", "Available", "Traffic Intelligence"],
              ].map(([title, description, status, page]) => (
                <article className="safety-module-card" key={title}>
                  <span className="module-status available">{status}</span>
                  <h3>{title}</h3>
                  <p>{description}</p>
<button type="button" onClick={() => setActivePage(page)}>
  Open module <ArrowRight size={15} />
</button>                </article>
              ))}
            </div>
          </section>
        )}

        {activePage === "Industrial Safety" && (
          <section className="safety-page">
            <div className="safety-page-heading">
              <span>INDUSTRIAL SAFETY INTELLIGENCE</span>
              <h2>Workplace Safety Command Center</h2>
              <p>Plan AI-powered monitoring for PPE compliance, fire risks and worker safety.</p>
            </div>
            <div className="safety-overview">
              <div><strong>05</strong><span>Planned detection modules</span></div>
              <div><strong>0</strong><span>Connected industrial cameras</span></div>
              <div><strong>--</strong><span>Verified safety alerts</span></div>
            </div>
            <div className="safety-module-grid">
              
              {[
                ["PPE Detection", "Helmet, gloves, goggles and safety vest compliance.", "PPE Detection", <HardHat size={24} />],
                ["Fire & Smoke Detection", "Identify potential fire and smoke events.", "Fire & Smoke Detection", <Flame size={24} />],
                ["Worker Safety", "Monitor designated workplace safety conditions.", "Worker Safety", <UserRound size={24} />],
              ].map(([title, description, page, icon]) => (
                <article className="safety-module-card" key={title}>
                  <span className="safety-module-icon">{icon}</span>
                  <span className="module-status planned">Coming Soon</span>
                  <h3>{title}</h3>
                  <p>{description}</p>
<button type="button" onClick={() => setActivePage(page)}>
  View module <ArrowRight size={15} />
</button>                </article>
              ))}
            </div>
          </section>
        )}

        {activePage === "PPE Detection" && <PpeDetection />}

      {activePage === "Fire & Smoke Detection" && <FireSmokeDetection />}

        {activePage === "Worker Safety" && (
          <section className="safety-page">
            <div className="safety-page-heading">
              <span>WORKFORCE PROTECTION</span>
              <h2>Worker Safety Monitoring</h2>
              <p>Future monitoring of workplace safety conditions and PPE compliance.</p>
            </div>
            <div className="safety-module-grid">
              {[
                ["PPE Compliance", "Check required protective equipment."],
                ["Restricted Area Monitoring", "Identify potential entry into designated restricted zones."],
                ["Safety Event Review", "Review and verify detected safety events."],
              ].map(([title, description]) => (
                <article className="safety-module-card" key={title}>
                  <span className="module-status planned">Coming Soon</span>
                  <h3>{title}</h3>
                  <p>{description}</p>
                </article>
              ))}
            </div>
          </section>
        )}
        {activePage === "AI Assistant" && <section className="page-content">

          <h2> AISDS Assistant</h2><p>Ask questions about your current traffic analysis.</p>

          <div className="assistant-suggestions">

            {["How many vehicles were detected?", "What is the congestion level?", "How many line crossings?", "What is the average speed?", "Show analysis history"].map((question) =>

              <button key={question} type="button" onClick={() => setAssistantQuestion(question)}>{question}</button>)}

          </div>

          <form className="assistant-form" onSubmit={handleAssistantAsk}>

            <input type="text" value={assistantQuestion} onChange={(event) => setAssistantQuestion(event.target.value)} placeholder="Ask a traffic-related question..." />

            <button type="submit" disabled={!assistantQuestion.trim()}>Ask</button>

          </form>

          <div className="assistant-messages">

            {assistantMessages.length === 0 ? <p>Your traffic questions and answers will appear here.</p> :

              [...assistantMessages].reverse().map((message, index) => <article className="assistant-message" key={`${message.question}-${index}`}>

                <p><strong>You:</strong> {message.question}</p><p><strong>AISDS:</strong> {message.answer}</p>

              </article>)}

          </div>

        </section>}

      </main>

    </div>

  );

}

export default App;
















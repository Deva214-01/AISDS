
import { useEffect, useState } from "react";
import "./TrafficIntelligence.css";

const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8001";

export default function TrafficIntelligence({
  cameraEnabled,
  cameraStats,
  cameraError,
  onStartCamera,
  onStopCamera,
}) {
  const [stats, setStats] = useState(null);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    let active = true;

    const loadData = async () => {
      try {
        const [statsResponse, alertsResponse] = await Promise.all([
          fetch(`${API}/api/camera/stats`),
          fetch(`${API}/api/camera/helmet-alerts`),
        ]);

        if (!statsResponse.ok) {
          throw new Error("Camera statistics unavailable");
        }

        const statsData = await statsResponse.json();
        const alertsData = alertsResponse.ok
          ? await alertsResponse.json()
          : { alerts: [] };

        if (!active) return;

        setStats(statsData);
        setAlerts(alertsData.alerts || []);
      } catch (error) {
        console.error("Helmet Detection polling error:", error);
      }
    };

    loadData();
    const interval = setInterval(loadData, 2000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const currentStats = cameraStats || stats;
  const cameraRunning =
    cameraEnabled && currentStats?.status === "running";

  const metrics = [
    {
      icon: "🚘",
      label: "Vehicles detected",
      value: currentStats?.total_vehicles ?? 0,
      description: "Latest camera frame",
      color: "cyan",
    },
    {
      icon: "🏍️",
      label: "Motorcycles",
      value: currentStats?.motorcycles ?? 0,
      description: "Latest camera frame",
      color: "purple",
    },
    {
      icon: "🛣️",
      label: "Unique motorcycles",
      value: currentStats?.session_motorcycles ?? 0,
      description: "Current session",
      color: "lime",
    },
    {
      icon: "🚨",
      label: "Helmet alerts",
      value: alerts.length,
      description: "Reported violations",
      color: "red",
    },
  ];

  return (
    <main className="page-content ti-page">

      {/* HERO HEADER */}
      <header className="helmet-hero">
        <div className="helmet-hero-content">
          <div className="helmet-brand">
            <span className="helmet-brand-icon">🪖</span>
            <span>TRAFFICX <b>AI</b></span>
            <span className="helmet-brand-divider">/</span>
            <span className="helmet-brand-section">SAFETY SYSTEM</span>
          </div>

          <div className="helmet-heading">
            <span className="ti-eyebrow">INTELLIGENT ROAD SAFETY · MODULE 01</span>
            <h1>Helmet <span>Detection</span></h1>
            <p>
              AI-powered rider monitoring. Detect unsafe riding,
              identify helmet violations and improve road safety.
            </p>
          </div>

          <div className="helmet-hero-bottom">
            <div className="helmet-feature">
              <span>📡</span>
              <span>REAL-TIME MONITORING</span>
            </div>
            <div className="helmet-feature">
              <span>🤖</span>
              <span>AI COMPUTER VISION</span>
            </div>
            <div className="helmet-feature">
              <span>🛡️</span>
              <span>ROAD SAFETY</span>
            </div>
          </div>
        </div>

        <div className="helmet-hero-visual">
          <div className="helmet-orbit orbit-one"></div>
          <div className="helmet-orbit orbit-two"></div>
          <div className="helmet-center-icon">🪖</div>
          <span className="helmet-orbit-label">AI ACTIVE</span>
        </div>
      </header>

      {/* CAMERA STATUS */}
      <section className="helmet-toolbar">
        <div className="helmet-toolbar-left">
          <span className={`helmet-status-light ${cameraRunning ? "active" : ""}`}></span>
          <div>
            <strong>{cameraRunning ? "Monitoring active" : "Monitoring offline"}</strong>
            <span>
              {cameraRunning
                ? "Camera connected · Processing live frames"
                : "Camera is waiting for activation"}
            </span>
          </div>
        </div>

        <div className="helmet-toolbar-right">
          <span className="helmet-system-tag">SYSTEM STATUS</span>
          <span className={`helmet-status-pill ${cameraRunning ? "online" : "offline"}`}>
            {cameraRunning ? "● ONLINE" : "● OFFLINE"}
          </span>
        </div>
      </section>

      {cameraError && (
        <div className="ti-error">
          <span>⚠️</span> {cameraError}
        </div>
      )}

      {/* METRICS */}
      <section className="helmet-section">
        <div className="helmet-section-heading">
          <div>
            <span className="ti-eyebrow">LIVE ANALYTICS</span>
            <h2>Detection Overview <span>📊</span></h2>
          </div>
          <span className="helmet-updated">↻ AUTO REFRESH · 2 SEC</span>
        </div>

        <div className="helmet-metrics">
          {metrics.map((metric) => (
            <article className={`helmet-metric ${metric.color}`} key={metric.label}>
              <div className="helmet-metric-top">
                <span className="helmet-metric-icon">{metric.icon}</span>
                <span className="helmet-metric-indicator">LIVE</span>
              </div>
              <strong>{metric.value}</strong>
              <span className="helmet-metric-label">{metric.label}</span>
              <span className="helmet-metric-description">{metric.description}</span>
            </article>
          ))}
        </div>
      </section>

      {/* MAIN MONITORING AREA */}
      <section className="helmet-main-grid">

        {/* LIVE CAMERA */}
        <article className="helmet-camera-panel">
          <div className="helmet-panel-heading">
            <div>
              <span className="ti-eyebrow">VISION ENGINE / 01</span>
              <h2>Live Camera Feed <span>📹</span></h2>
              <p>Visual monitoring and AI detection output</p>
            </div>

            <span className={`helmet-live-badge ${cameraRunning ? "recording" : ""}`}>
              <span></span>
              {cameraRunning ? "LIVE FEED" : "STANDBY"}
            </span>
          </div>

          <div className="helmet-camera-screen">
            {cameraEnabled ? (
              <>
                <img
                  className="ti-video"
                  src={`${API}/api/camera/live`}
                  alt="Live helmet detection camera feed"
                />
                <div className="helmet-camera-overlay">
                  <span>📡 TRAFFICX VISION</span>
                  <span>AI MONITORING</span>
                </div>
              </>
            ) : (
              <div className="ti-placeholder helmet-placeholder">
                <span className="helmet-placeholder-icon">📹</span>
                <strong>Camera is waiting</strong>
                <p>Activate your camera to start AI-powered helmet detection.</p>
                <span className="helmet-placeholder-hint">READY TO INITIALIZE</span>
              </div>
            )}
          </div>

          <div className="helmet-camera-footer">
            <div className="helmet-camera-info">
              <span className="helmet-camera-info-icon">🎯</span>
              <div>
                <strong>Detection engine</strong>
                <span>YOLO-based vehicle and helmet analysis</span>
              </div>
            </div>

            {!cameraEnabled ? (
              <button
                type="button"
                className="helmet-action-button start"
                onClick={onStartCamera}
              >
                <span>▶</span> Start Monitoring
              </button>
            ) : (
              <button
                type="button"
                className="helmet-action-button stop"
                onClick={onStopCamera}
              >
                <span>■</span> Stop Monitoring
              </button>
            )}
          </div>
        </article>

        {/* ALERT CENTER */}
        <article className="helmet-alert-panel">
          <div className="helmet-panel-heading">
            <div>
              <span className="ti-eyebrow">SAFETY INTELLIGENCE / 02</span>
              <h2>Violation Center <span>🚨</span></h2>
              <p>Helmet safety alerts from the backend</p>
            </div>
          </div>

          <div className="helmet-alert-summary">
            <div className="helmet-alert-summary-icon">⚠️</div>
            <div>
              <strong>{alerts.length}</strong>
              <span>Reported helmet alerts</span>
            </div>
          </div>

          <div className="helmet-alert-content">
            {alerts.length === 0 ? (
              <div className="helmet-empty">
                <span>🛡️</span>
                <strong>No active alerts</strong>
                <p>
                  No helmet violations have been reported by the detection
                  system yet.
                </p>
                <small>SAFETY MONITORING READY</small>
              </div>
            ) : (
              <div className="ti-alert-list">
                {alerts.slice(0, 10).map((alert, index) => (
                  <div className="helmet-alert-item" key={alert.id || index}>
                    <span className="helmet-alert-icon">🚨</span>
                    <div className="helmet-alert-details">
                      <strong>Possible No-Helmet Violation</strong>
                      <span>{alert.timestamp || "Time unavailable"}</span>
                    </div>
                    <span className="helmet-alert-number">
                      #{index + 1}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="helmet-alert-footer">
            <span>🔔 ALERT FEED</span>
            <span>MAX 10 RECENT EVENTS</span>
          </div>
        </article>
      </section>

      {/* SAFETY INFORMATION */}
      <section className="helmet-safety-strip">
        <div className="helmet-safety-icon">💡</div>
        <div>
          <span className="ti-eyebrow">ROAD SAFETY INSIGHT</span>
          <h3>Every ride deserves protection.</h3>
          <p>
            Helmet detection helps identify potential safety violations.
            AI-generated alerts should be reviewed before enforcement.
          </p>
        </div>
        <div className="helmet-safety-emoji">🏍️</div>
      </section>

      <footer className="helmet-footer">
        <span>🪖 TRAFFICX AI · HELMET DETECTION MODULE</span>
        <span>BUILT FOR SMARTER & SAFER ROADS</span>
      </footer>
    </main>
  );
}


import { useEffect, useRef, useState } from "react";

const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8001";

export default function BrowserCamera({ enabled }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [annotatedFrame, setAnnotatedFrame] = useState("");
  const [cameraError, setCameraError] = useState("");

  useEffect(() => {
    if (!enabled) {
      setAnnotatedFrame("");
      setCameraError("");
      return undefined;
    }

    let active = true;
    let stream = null;
    let timer = null;
    let inFlight = false;

    const start = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("Browser camera requires HTTPS and camera permission.");
        }

        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "environment"
          },
          audio: false
        });

        if (!active) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        const video = videoRef.current;
        if (!video) return;

        video.srcObject = stream;
        await video.play();

        const sendFrame = () => {
          if (!active || inFlight || !videoRef.current || videoRef.current.readyState < 2) {
            timer = setTimeout(sendFrame, 500);
            return;
          }

          inFlight = true;
          const canvas = canvasRef.current;
          const currentVideo = videoRef.current;
          canvas.width = 640;
          canvas.height = 480;

          const context = canvas.getContext("2d");
          context.drawImage(currentVideo, 0, 0, canvas.width, canvas.height);

          canvas.toBlob(async (blob) => {
            try {
              if (!active || !blob) return;

              const formData = new FormData();
              formData.append("file", blob, "browser-camera.jpg");

              const response = await fetch(`${API}/api/camera/frame`, {
                method: "POST",
                body: formData
              });

              if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Camera API error ${response.status}`);
              }

              const data = await response.json();
              if (active && data.image) {
                setAnnotatedFrame(`data:image/jpeg;base64,${data.image}`);
                setCameraError("");
              }
            } catch (error) {
              if (active) setCameraError(error.message || "Frame processing failed.");
            } finally {
              inFlight = false;
              if (active) timer = setTimeout(sendFrame, 700);
            }
          }, "image/jpeg", 0.75);
        };

        sendFrame();
      } catch (error) {
        if (active) setCameraError(error.message || "Could not start laptop camera.");
      }
    };

    start();

    return () => {
      active = false;
      if (timer) clearTimeout(timer);
      if (stream) stream.getTracks().forEach((track) => track.stop());
      if (videoRef.current) videoRef.current.srcObject = null;

      fetch(`${API}/api/camera/stop`, {
        method: "POST",
        keepalive: true
      }).catch(() => {});
    };
  }, [enabled]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative", overflow: "hidden" }}>
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="ti-video"
        style={{ display: "block", width: "100%", height: "100%", objectFit: "cover" }}
      />
      <canvas ref={canvasRef} style={{ display: "none" }} />
      {annotatedFrame && (
        <img
          src={annotatedFrame}
          alt="AI annotated camera preview"
          style={{
            position: "absolute",
            right: 10,
            bottom: 10,
            width: "34%",
            height: "auto",
            maxHeight: "34%",
            objectFit: "contain",
            border: "2px solid #22c55e",
            borderRadius: 8,
            background: "#111"
          }}
        />
      )}
      {cameraError && (
        <p style={{ color: "#ff7777", padding: 8 }}>{cameraError}</p>
      )}
    </div>
  );
}

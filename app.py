import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components


try:
    # Your real project can provide `model.predict(x)` with scores 1..5.
    from model import model  # type: ignore
except Exception:
    from model_stub import model  # type: ignore


@dataclass
class LiveReadings:
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    t_ms: int = 0


def _init_state() -> None:
    st.session_state.setdefault("riding", False)
    st.session_state.setdefault("scores", [])
    st.session_state.setdefault("latest", LiveReadings().__dict__)
    st.session_state.setdefault("final_rating", None)
    st.session_state.setdefault("last_window_n", None)
    st.session_state.setdefault("last_score", None)


def _sensor_component(
    *,
    riding: bool,
    sample_hz: int = 50,
    window_sec: int = 5,
    break_sec: int = 2,
    live_send_hz: int = 5,
    key: str = "sensors",
) -> Optional[Dict[str, Any]]:
    """
    Browser-side DeviceMotion collector.

    Sends messages back to Streamlit as JSON objects:
      - {"type":"live","t_ms":..., "ax":..,"ay":..,"az":..,"gx":..,"gy":..,"gz":..}
      - {"type":"window","t_ms":..., "samples":[[ax,ay,az,gx,gy,gz], ...]}
    """
    cfg = {
        "riding": bool(riding),
        "sampleHz": int(sample_hz),
        "windowSec": int(window_sec),
        "breakSec": int(break_sec),
        "liveSendHz": int(live_send_hz),
    }
    html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 12px; }}
      .row {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
      .pill {{ padding: 6px 10px; border-radius: 999px; background: #f3f4f6; }}
      button {{ padding: 8px 10px; border-radius: 10px; border: 1px solid #e5e7eb; background: white; cursor: pointer; }}
      button:disabled {{ opacity: 0.6; cursor: not-allowed; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
      .muted {{ color: #6b7280; }}
    </style>
  </head>
  <body>
    <div class="row">
      <button id="btnEnable">Enable sensors</button>
      <span id="status" class="pill">status: idle</span>
      <span class="pill mono">mode: <span id="mode">stopped</span></span>
      <span class="pill mono">hz: <span id="hz">0</span></span>
    </div>
    <div style="height: 10px;"></div>
    <div class="muted">Tip: open this app on your phone (Safari/iOS needs “Enable sensors”). Keep the phone screen on during capture.</div>

    <script>
      const CFG = {json.dumps(cfg)};

      // Streamlit custom component messaging without a full component build.
      function sendToStreamlit(valueObj) {{
        const msg = {{
          isStreamlitMessage: true,
          type: "streamlit:setComponentValue",
          value: valueObj
        }};
        window.parent.postMessage(msg, "*");
      }}

      const els = {{
        btnEnable: document.getElementById("btnEnable"),
        status: document.getElementById("status"),
        mode: document.getElementById("mode"),
        hz: document.getElementById("hz"),
      }};

      function setStatus(text) {{
        els.status.textContent = "status: " + text;
      }}
      function setMode(text) {{
        els.mode.textContent = text;
      }}

      let permissionGranted = false;
      let latestAccel = {{x:0,y:0,z:0}};
      let latestGyro = {{x:0,y:0,z:0}};
      let lastMotionAt = 0;

      function nowMs() {{ return Date.now(); }}

      function clampNum(v) {{
        if (v === null || v === undefined) return 0;
        const n = Number(v);
        if (!Number.isFinite(n)) return 0;
        return n;
      }}

      function onMotion(e) {{
        lastMotionAt = nowMs();

        // Accelerometer
        const a = e.accelerationIncludingGravity || e.acceleration || {{}};
        latestAccel = {{
          x: clampNum(a.x),
          y: clampNum(a.y),
          z: clampNum(a.z),
        }};

        // Gyroscope (rotationRate: alpha,beta,gamma) -> map to x,y,z for model.
        const r = e.rotationRate || {{}};
        latestGyro = {{
          x: clampNum(r.beta),   // pitch rate
          y: clampNum(r.gamma),  // roll rate
          z: clampNum(r.alpha),  // yaw rate
        }};
      }}

      async function requestPermissionIfNeeded() {{
        try {{
          if (typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function") {{
            const res = await DeviceMotionEvent.requestPermission();
            if (res !== "granted") throw new Error("permission not granted");
          }}
          permissionGranted = true;
          setStatus("permission granted");
          return true;
        }} catch (err) {{
          permissionGranted = false;
          setStatus("permission denied / unavailable");
          return false;
        }}
      }}

      let sampleTimer = null;
      let liveTimer = null;
      let breakTimer = null;
      let windowTimer = null;
      let windowStartMs = 0;
      let windowSamples = [];
      let sampleCount = 0;
      let lastHzUpdateMs = 0;

      function clearTimers() {{
        if (sampleTimer) clearInterval(sampleTimer);
        if (liveTimer) clearInterval(liveTimer);
        if (breakTimer) clearTimeout(breakTimer);
        if (windowTimer) clearTimeout(windowTimer);
        sampleTimer = null; liveTimer = null; breakTimer = null; windowTimer = null;
      }}

      function startCaptureLoop() {{
        clearTimers();
        windowSamples = [];
        sampleCount = 0;
        windowStartMs = nowMs();
        lastHzUpdateMs = windowStartMs;
        setMode("capturing");
        setStatus("capturing window");

        const sampleEveryMs = Math.max(5, Math.round(1000 / CFG.sampleHz));
        const liveEveryMs = Math.max(50, Math.round(1000 / CFG.liveSendHz));

        // Sample at ~50Hz using latest event readings (events can be irregular).
        sampleTimer = setInterval(() => {{
          const t = nowMs();
          const row = [
            latestAccel.x, latestAccel.y, latestAccel.z,
            latestGyro.x, latestGyro.y, latestGyro.z
          ];
          windowSamples.push(row);
          sampleCount += 1;

          // Update a simple hz estimate once per second.
          if (t - lastHzUpdateMs >= 1000) {{
            const elapsedS = (t - windowStartMs) / 1000;
            const hz = elapsedS > 0 ? (sampleCount / elapsedS) : 0;
            els.hz.textContent = hz.toFixed(1);
            lastHzUpdateMs = t;
          }}
        }}, sampleEveryMs);

        // Send live readings at a low frequency to avoid too many reruns.
        liveTimer = setInterval(() => {{
          sendToStreamlit({{
            type: "live",
            t_ms: nowMs(),
            ax: latestAccel.x, ay: latestAccel.y, az: latestAccel.z,
            gx: latestGyro.x,  gy: latestGyro.y,  gz: latestGyro.z
          }});
        }}, liveEveryMs);

        // End window after windowSec; then send batch; then break.
        windowTimer = setTimeout(() => {{
          setStatus("sending window");
          sendToStreamlit({{
            type: "window",
            t_ms: nowMs(),
            samples: windowSamples
          }});
          clearTimers();
          setMode("break");
          setStatus("break");
          breakTimer = setTimeout(() => {{
            if (CFG.riding) startCaptureLoop();
          }}, Math.max(0, CFG.breakSec) * 1000);
        }}, Math.max(1, CFG.windowSec) * 1000);
      }}

      function stopCaptureLoop() {{
        clearTimers();
        setMode("stopped");
        setStatus(permissionGranted ? "stopped" : "idle");
        els.hz.textContent = "0";
      }}

      function attachListeners() {{
        window.removeEventListener("devicemotion", onMotion);
        window.addEventListener("devicemotion", onMotion, {{ passive: true }});
      }}

      els.btnEnable.addEventListener("click", async () => {{
        setStatus("requesting permission");
        const ok = await requestPermissionIfNeeded();
        if (ok) {{
          attachListeners();
          setStatus("listening");
          if (CFG.riding) startCaptureLoop();
        }}
      }});

      // Auto-attach listener when permission isn't required (Android/Chrome).
      (function init() {{
        setMode(CFG.riding ? "starting" : "stopped");
        if (typeof DeviceMotionEvent === "undefined") {{
          setStatus("DeviceMotionEvent not supported");
          els.btnEnable.disabled = true;
          return;
        }}

        if (typeof DeviceMotionEvent.requestPermission !== "function") {{
          permissionGranted = true;
          attachListeners();
          setStatus("listening");
          if (CFG.riding) startCaptureLoop();
        }} else {{
          setStatus("permission required");
        }}

        // Stop/start if Streamlit rerenders with new CFG.riding
        if (!CFG.riding) stopCaptureLoop();
      }})();

      // Basic "no events" detection.
      setInterval(() => {{
        if (!permissionGranted) return;
        const age = nowMs() - lastMotionAt;
        if (age > 2000) setStatus("no motion events (screen locked?)");
      }}, 1000);
    </script>
  </body>
</html>
"""
    return components.html(html, height=140, key=key)


def _parse_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            obj = json.loads(payload)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _predict_window(samples: List[List[float]]) -> Tuple[Optional[int], Optional[int]]:
    if not samples:
        return None, 0
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 6:
        return None, int(x.shape[0]) if x.ndim >= 1 else 0
    score = model.predict(x)
    try:
        score_int = int(round(float(score)))
    except Exception:
        return None, int(x.shape[0])
    score_int = max(1, min(5, score_int))
    return score_int, int(x.shape[0])


def main() -> None:
    st.set_page_config(page_title="Ride Rating (Sensors → NN)", layout="wide")
    _init_state()

    st.title("Ride Rating (Mobile Sensors → Neural Net)")

    left, right = st.columns([1.1, 1.4], gap="large")
    with left:
        st.subheader("Controls")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Start Ride", type="primary", use_container_width=True, disabled=st.session_state.riding):
                st.session_state.riding = True
                st.session_state.scores = []
                st.session_state.final_rating = None
                st.session_state.last_window_n = None
                st.session_state.last_score = None
        with c2:
            if st.button("Stop Ride", use_container_width=True, disabled=not st.session_state.riding):
                st.session_state.riding = False
                if st.session_state.scores:
                    st.session_state.final_rating = float(np.mean(st.session_state.scores))
                else:
                    st.session_state.final_rating = None

        st.caption("Open this app on your phone to capture motion sensors.")

        st.subheader("Live readings")
        live_placeholder = st.empty()

        st.subheader("Scores during ride")
        scores_placeholder = st.empty()

        st.subheader("Final Ride Rating")
        if st.session_state.final_rating is None:
            st.info("Stop the ride to see the final rating.")
        else:
            st.success(f"Final Ride Rating: {st.session_state.final_rating:.2f} / 5.00")

    with right:
        st.subheader("Sensor capture")
        st.write(
            "Enable sensors in the embedded panel. While riding, the browser collects ~50Hz samples, "
            "sends a **5-second window**, then pauses **2 seconds**, and repeats."
        )

        payload_raw = _sensor_component(
            riding=st.session_state.riding,
            sample_hz=50,
            window_sec=5,
            break_sec=2,
            live_send_hz=5,
            key="sensor_panel",
        )
        payload = _parse_payload(payload_raw)

        if payload and isinstance(payload.get("type"), str):
            ptype = payload["type"]
            if ptype == "live":
                st.session_state.latest = {
                    "ax": float(payload.get("ax", 0.0)),
                    "ay": float(payload.get("ay", 0.0)),
                    "az": float(payload.get("az", 0.0)),
                    "gx": float(payload.get("gx", 0.0)),
                    "gy": float(payload.get("gy", 0.0)),
                    "gz": float(payload.get("gz", 0.0)),
                    "t_ms": int(payload.get("t_ms", 0)),
                }
            elif ptype == "window" and st.session_state.riding:
                samples = payload.get("samples", [])
                if isinstance(samples, list):
                    score, n = _predict_window(samples)
                    st.session_state.last_window_n = n
                    st.session_state.last_score = score
                    if score is not None:
                        st.session_state.scores.append(score)

        # Update left-side live widgets from session_state (no loops/sleeps).
        latest = st.session_state.latest
        live_placeholder.dataframe(
            {
                "accel_x": [latest.get("ax", 0.0)],
                "accel_y": [latest.get("ay", 0.0)],
                "accel_z": [latest.get("az", 0.0)],
                "gyro_x": [latest.get("gx", 0.0)],
                "gyro_y": [latest.get("gy", 0.0)],
                "gyro_z": [latest.get("gz", 0.0)],
                "t_ms": [latest.get("t_ms", 0)],
            },
            hide_index=True,
            use_container_width=True,
        )

        if st.session_state.scores:
            scores_placeholder.write(
                {
                    "count": len(st.session_state.scores),
                    "scores": st.session_state.scores,
                    "mean_so_far": float(np.mean(st.session_state.scores)),
                    "last_score": st.session_state.last_score,
                    "last_window_samples": st.session_state.last_window_n,
                }
            )
        else:
            scores_placeholder.write(
                {
                    "count": 0,
                    "scores": [],
                    "last_score": st.session_state.last_score,
                    "last_window_samples": st.session_state.last_window_n,
                }
            )


if __name__ == "__main__":
    main()


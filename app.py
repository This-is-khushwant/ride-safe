import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components


try:
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
    # Each entry: {"time": "HH:MM:SS", "score": int, "window": int}
    st.session_state.setdefault("score_entries", [])
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
    <div class="muted">Tip: open this app on your phone (Safari/iOS needs "Enable sensors"). Keep the phone screen on during capture.</div>

    <script>
      const CFG = {json.dumps(cfg)};

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

      function setStatus(text) {{ els.status.textContent = "status: " + text; }}
      function setMode(text)   {{ els.mode.textContent = text; }}

      let permissionGranted = false;
      let latestAccel = {{x:0,y:0,z:0}};
      let latestGyro  = {{x:0,y:0,z:0}};
      let lastMotionAt = 0;

      function nowMs() {{ return Date.now(); }}

      function clampNum(v) {{
        if (v === null || v === undefined) return 0;
        const n = Number(v);
        return Number.isFinite(n) ? n : 0;
      }}

      function onMotion(e) {{
        lastMotionAt = nowMs();
        const a = e.accelerationIncludingGravity || e.acceleration || {{}};
        latestAccel = {{ x: clampNum(a.x), y: clampNum(a.y), z: clampNum(a.z) }};
        const r = e.rotationRate || {{}};
        latestGyro  = {{ x: clampNum(r.beta), y: clampNum(r.gamma), z: clampNum(r.alpha) }};
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

      let sampleTimer = null, liveTimer = null, breakTimer = null, windowTimer = null;
      let windowStartMs = 0, windowSamples = [], sampleCount = 0, lastHzUpdateMs = 0;

      function clearTimers() {{
        if (sampleTimer)  clearInterval(sampleTimer);
        if (liveTimer)    clearInterval(liveTimer);
        if (breakTimer)   clearTimeout(breakTimer);
        if (windowTimer)  clearTimeout(windowTimer);
        sampleTimer = liveTimer = breakTimer = windowTimer = null;
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
        const liveEveryMs   = Math.max(50, Math.round(1000 / CFG.liveSendHz));

        sampleTimer = setInterval(() => {{
          const t = nowMs();
          windowSamples.push([
            latestAccel.x, latestAccel.y, latestAccel.z,
            latestGyro.x,  latestGyro.y,  latestGyro.z
          ]);
          sampleCount += 1;
          if (t - lastHzUpdateMs >= 1000) {{
            const elapsedS = (t - windowStartMs) / 1000;
            els.hz.textContent = elapsedS > 0 ? (sampleCount / elapsedS).toFixed(1) : "0";
            lastHzUpdateMs = t;
          }}
        }}, sampleEveryMs);

        liveTimer = setInterval(() => {{
          sendToStreamlit({{
            type: "live", t_ms: nowMs(),
            ax: latestAccel.x, ay: latestAccel.y, az: latestAccel.z,
            gx: latestGyro.x,  gy: latestGyro.y,  gz: latestGyro.z
          }});
        }}, liveEveryMs);

        windowTimer = setTimeout(() => {{
          setStatus("sending window");
          sendToStreamlit({{ type: "window", t_ms: nowMs(), samples: windowSamples }});
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
        if (!CFG.riding) stopCaptureLoop();
      }})();

      setInterval(() => {{
        if (!permissionGranted) return;
        if (nowMs() - lastMotionAt > 2000) setStatus("no motion events (screen locked?)");
      }}, 1000);
    </script>
  </body>
</html>
"""
    # ── FIX: removed `key=key` — not supported in older Streamlit versions ──
    return components.html(html, height=140)


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


def _predict_window(samples: List[List[float]]) -> Tuple[Optional[int], int]:
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


def _score_color(score: int) -> str:
    """Return a hex colour that goes green→yellow→red for scores 5→3→1."""
    colors = {5: "#22c55e", 4: "#84cc16", 3: "#eab308", 2: "#f97316", 1: "#ef4444"}
    return colors.get(score, "#6b7280")


def _render_score_cards(entries: List[Dict]) -> None:
    """Render a card for each scored window."""
    if not entries:
        st.info("No windows scored yet. Start riding to collect data.")
        return

    # Show newest first
    for i, entry in enumerate(reversed(entries)):
        score = entry["score"]
        color = _score_color(score)
        stars = "★" * score + "☆" * (5 - score)
        window_num = len(entries) - i

        st.markdown(
            f"""
            <div style="
                border-left: 5px solid {color};
                background: #f9fafb;
                border-radius: 10px;
                padding: 12px 16px;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <div>
                    <div style="font-weight: 600; font-size: 0.95rem; color: #111;">
                        Window #{window_num} &nbsp;·&nbsp; {entry['time']}
                    </div>
                    <div style="color: #6b7280; font-size: 0.82rem;">
                        {entry['samples']} samples
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size: 1.5rem; color: {color};">{score} / 5</div>
                    <div style="color: {color}; font-size: 0.9rem;">{stars}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="Ride Rating", layout="wide")
    _init_state()

    st.title("🏍️ Ride Rating — Mobile Sensors → Neural Net")

    left, right = st.columns([1.1, 1.4], gap="large")

    with left:
        st.subheader("Controls")
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "▶ Start Ride",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.riding,
            ):
                st.session_state.riding = True
                st.session_state.score_entries = []
                st.session_state.final_rating = None
                st.session_state.last_window_n = None
                st.session_state.last_score = None
        with c2:
            if st.button(
                "⏹ Stop Ride",
                use_container_width=True,
                disabled=not st.session_state.riding,
            ):
                st.session_state.riding = False
                scores = [e["score"] for e in st.session_state.score_entries]
                st.session_state.final_rating = float(np.mean(scores)) if scores else None

        st.caption("Open this app on your phone to capture motion sensors.")

        # ── Final rating banner ──
        if st.session_state.final_rating is not None:
            fr = st.session_state.final_rating
            color = _score_color(int(round(fr)))
            st.markdown(
                f"""
                <div style="
                    background: {color}22;
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 16px;
                    text-align: center;
                    margin-top: 12px;
                ">
                    <div style="font-size: 1.1rem; font-weight: 600; color: {color};">
                        Final Ride Rating
                    </div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: {color};">
                        {fr:.2f} / 5
                    </div>
                    <div style="color: {color}; font-size: 1.3rem;">
                        {"★" * int(round(fr)) + "☆" * (5 - int(round(fr)))}
                    </div>
                    <div style="color: #6b7280; font-size: 0.85rem; margin-top: 4px;">
                        based on {len(st.session_state.score_entries)} window(s)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif st.session_state.riding:
            scores = [e["score"] for e in st.session_state.score_entries]
            if scores:
                mean_so_far = float(np.mean(scores))
                st.info(f"Running mean: **{mean_so_far:.2f} / 5** ({len(scores)} windows so far)")
            else:
                st.info("Waiting for first window…")

        # ── Live readings ──
        st.subheader("Live readings")
        live_placeholder = st.empty()

        # ── Sensor data display ──
        latest = st.session_state.latest
        live_placeholder.dataframe(
            {
                "accel_x": [latest.get("ax", 0.0)],
                "accel_y": [latest.get("ay", 0.0)],
                "accel_z": [latest.get("az", 0.0)],
                "gyro_x":  [latest.get("gx", 0.0)],
                "gyro_y":  [latest.get("gy", 0.0)],
                "gyro_z":  [latest.get("gz", 0.0)],
            },
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.subheader("Sensor capture")
        st.write(
            "Enable sensors in the panel below. While riding, the browser collects ~50 Hz samples, "
            "sends a **5-second window**, pauses **2 seconds**, then repeats."
        )

        # ── THE FIX: no `key=` argument ──
        payload_raw = _sensor_component(
            riding=st.session_state.riding,
            sample_hz=50,
            window_sec=5,
            break_sec=2,
            live_send_hz=5,
        )
        payload = _parse_payload(payload_raw)

        if payload and isinstance(payload.get("type"), str):
            ptype = payload["type"]
            if ptype == "live":
                st.session_state.latest = {
                    "ax":  float(payload.get("ax", 0.0)),
                    "ay":  float(payload.get("ay", 0.0)),
                    "az":  float(payload.get("az", 0.0)),
                    "gx":  float(payload.get("gx", 0.0)),
                    "gy":  float(payload.get("gy", 0.0)),
                    "gz":  float(payload.get("gz", 0.0)),
                    "t_ms": int(payload.get("t_ms", 0)),
                }
            elif ptype == "window" and st.session_state.riding:
                samples = payload.get("samples", [])
                if isinstance(samples, list):
                    score, n = _predict_window(samples)
                    st.session_state.last_window_n = n
                    st.session_state.last_score = score
                    if score is not None:
                        st.session_state.score_entries.append(
                            {
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "score": score,
                                "samples": n,
                            }
                        )

        # ── Score cards ──
        st.subheader("Window scores")
        _render_score_cards(st.session_state.score_entries)


if __name__ == "__main__":
    main()
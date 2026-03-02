import json
import os
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

# ── Declare two-way custom component ─────────────────────────────────────────
_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensor_component")
_sensor_raw = components.declare_component("sensor_panel", path=_COMPONENT_DIR)

def _sensor_component(*, riding, sample_hz=50, window_sec=5, break_sec=2, live_send_hz=5):
    return _sensor_raw(
        riding=bool(riding),
        sample_hz=int(sample_hz),
        window_sec=int(window_sec),
        break_sec=int(break_sec),
        live_send_hz=int(live_send_hz),
        key="sensor_panel",
        default=None,
    )

@dataclass
class LiveReadings:
    ax: float=0.0; ay: float=0.0; az: float=0.0
    gx: float=0.0; gy: float=0.0; gz: float=0.0
    t_ms: int=0

def _init_state():
    st.session_state.setdefault("riding", False)
    st.session_state.setdefault("score_entries", [])
    st.session_state.setdefault("latest", LiveReadings().__dict__)
    st.session_state.setdefault("final_rating", None)
    st.session_state.setdefault("last_window_n", None)
    st.session_state.setdefault("last_score", None)
    st.session_state.setdefault("ride_start_time", None)

def _fmt_elapsed(start: float) -> str:
    total = max(0, int(time.time() - start))
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"

def _parse_payload(payload) -> Optional[Dict]:
    if payload is None: return None
    if isinstance(payload, dict): return payload
    if isinstance(payload, str):
        try:
            obj = json.loads(payload)
            return obj if isinstance(obj, dict) else None
        except: return None
    return None

def _predict_window(samples) -> Tuple[Optional[int], int]:
    if not samples: return None, 0
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 6:
        return None, int(x.shape[0]) if x.ndim >= 1 else 0
    try:
        score = model.predict(x)
        return max(1, min(5, int(round(float(score))))), int(x.shape[0])
    except: return None, int(x.shape[0])

def _color(score: int) -> str:
    return {5:"#22c55e",4:"#84cc16",3:"#eab308",2:"#f97316",1:"#ef4444"}.get(score,"#6b7280")

def _render_cards(entries):
    if not entries:
        st.info("No windows scored yet. Start riding to collect data.")
        return
    for i, e in enumerate(reversed(entries)):
        s=e["score"]; c=_color(s); stars="★"*s+"☆"*(5-s); w=len(entries)-i
        st.markdown(f"""
<div style="border-left:5px solid {c};background:#f9fafb;border-radius:10px;
  padding:12px 16px;margin-bottom:10px;display:flex;
  justify-content:space-between;align-items:center;">
  <div>
    <div style="font-weight:600;color:#111;">Window #{w} &nbsp;·&nbsp; {e['time']}</div>
    <div style="color:#6b7280;font-size:.82rem;">{e['samples']} samples</div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:1.5rem;color:{c};font-weight:700;">{s} / 5</div>
    <div style="color:{c};">{stars}</div>
  </div>
</div>""", unsafe_allow_html=True)

def _render_timer(start, riding):
    if start is None:
        st.markdown("""<div style="background:#f3f4f6;border-radius:12px;padding:14px;
          text-align:center;color:#9ca3af;">Timer starts when you begin a ride.</div>""",
          unsafe_allow_html=True)
        return
    elapsed = _fmt_elapsed(start)
    c = "#22c55e" if riding else "#6b7280"
    label = "Ride in progress" if riding else "Ride ended"
    st.markdown(f"""
<div style="background:{c}11;border:2px solid {c};border-radius:14px;
  padding:16px;text-align:center;margin-bottom:12px;">
  <div style="font-size:.85rem;color:{c};font-weight:600;">{label}</div>
  <div style="font-size:2.8rem;font-weight:800;color:{c};
    font-family:monospace;letter-spacing:2px;">{elapsed}</div>
  <div style="font-size:.78rem;color:#6b7280;margin-top:4px;">MM : SS</div>
</div>""", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Ride Rating", layout="wide")
    _init_state()
    st.title("Ride Rating — Mobile Sensors")

    left, right = st.columns([1.1, 1.4], gap="large")

    with left:
        st.subheader("Controls")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ Start Ride", type="primary", use_container_width=True,
                         disabled=st.session_state.riding):
                st.session_state.riding = True
                st.session_state.score_entries = []
                st.session_state.final_rating = None
                st.session_state.last_window_n = None
                st.session_state.last_score = None
                st.session_state.ride_start_time = time.time()
        with c2:
            if st.button("⏹ Stop Ride", use_container_width=True,
                         disabled=not st.session_state.riding):
                st.session_state.riding = False
                scores = [e["score"] for e in st.session_state.score_entries]
                st.session_state.final_rating = float(np.mean(scores)) if scores else None

        st.caption("Open this app on your phone to capture motion sensors.")

        st.subheader("Ride Timer")
        _render_timer(st.session_state.ride_start_time, st.session_state.riding)

        if st.session_state.final_rating is not None:
            fr = st.session_state.final_rating; c = _color(int(round(fr)))
            st.markdown(f"""
<div style="background:{c}22;border:2px solid {c};border-radius:12px;
  padding:16px;text-align:center;margin-top:10px;">
  <div style="font-size:1rem;font-weight:600;color:{c};">Final Ride Rating</div>
  <div style="font-size:2.5rem;font-weight:800;color:{c};">{fr:.2f} / 5</div>
  <div style="color:{c};font-size:1.2rem;">{"★"*int(round(fr))}{"☆"*(5-int(round(fr)))}</div>
  <div style="color:#6b7280;font-size:.82rem;margin-top:4px;">
    based on {len(st.session_state.score_entries)} window(s)</div>
</div>""", unsafe_allow_html=True)
        elif st.session_state.riding:
            scores = [e["score"] for e in st.session_state.score_entries]
            if scores:
                st.info(f"Running mean: **{float(np.mean(scores)):.2f} / 5** ({len(scores)} windows)")
            else:
                st.info("Waiting for first 5-second window...")

        st.subheader("Live Readings")
        latest = st.session_state.latest
        ax=latest.get('ax',0.0); ay=latest.get('ay',0.0); az=latest.get('az',0.0)
        gx=latest.get('gx',0.0); gy=latest.get('gy',0.0); gz=latest.get('gz',0.0)
        st.markdown(
            f"| Sensor | X | Y | Z |\n|--------|--:|--:|--:|\n"
            f"| **Accel** | `{ax:.3f}` | `{ay:.3f}` | `{az:.3f}` |\n"
            f"| **Gyro**  | `{gx:.3f}` | `{gy:.3f}` | `{gz:.3f}` |"
        )

    with right:
        st.subheader("Sensor Capture Panel")
        st.markdown(
            "> **Must be opened on a phone.** Desktop browsers do not expose DeviceMotion. "
            "Share the URL to your phone, then tap **Enable sensors**."
        )

        payload_raw = _sensor_component(
            riding=st.session_state.riding,
            sample_hz=50, window_sec=5, break_sec=2, live_send_hz=5,
        )
        payload = _parse_payload(payload_raw)

        if payload and isinstance(payload.get("type"), str):
            ptype = payload["type"]
            if ptype == "live":
                st.session_state.latest = {
                    "ax": float(payload.get("ax",0)), "ay": float(payload.get("ay",0)),
                    "az": float(payload.get("az",0)), "gx": float(payload.get("gx",0)),
                    "gy": float(payload.get("gy",0)), "gz": float(payload.get("gz",0)),
                    "t_ms": int(payload.get("t_ms",0)),
                }
                st.rerun()  # refreshes timer + live table at ~5 Hz
            elif ptype == "window" and st.session_state.riding:
                samples = payload.get("samples", [])
                if isinstance(samples, list):
                    score, n = _predict_window(samples)
                    st.session_state.last_window_n = n
                    st.session_state.last_score = score
                    if score is not None:
                        st.session_state.score_entries.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "score": score, "samples": n,
                        })

        st.subheader("Window Scores")
        _render_cards(st.session_state.score_entries)

if __name__ == "__main__":
    main()

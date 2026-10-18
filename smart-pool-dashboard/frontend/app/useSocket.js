"use client";
// Custom hook: single Socket.IO connection to the Python backend.
import { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";

const BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:5000";

export function useSocket() {
  const socketRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState(null);
  const [state, setState] = useState(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    const socket = io(BACKEND, { transports: ["websocket", "polling"] });
    socketRef.current = socket;

    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));
    socket.on("state_update", (s) => setState(s));
    socket.on("frame_update", (d) => {
      setFrame(`data:image/jpeg;base64,${d.frame}`);
      setState(d.state);
    });
    socket.on("detection_started", () => setRunning(true));
    socket.on("detection_stopped", () => setRunning(false));

    return () => socket.disconnect();
  }, []);

  return {
    connected,
    frame,
    state,
    running,
    start: () => socketRef.current?.emit("start_detection"),
    stop: () => socketRef.current?.emit("stop_detection"),
    startWaterTest: () => socketRef.current?.emit("start_water_test"),
  };
}

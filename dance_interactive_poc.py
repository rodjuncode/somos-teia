#!/usr/bin/env python3
"""PoC de danca interativa com Kinect v1 ou webcam.

Instalacao no Ubuntu (para Kinect, instale primeiro as libs nativas):
    sudo apt-get install libfreenect-dev freenect python3-dev python3-venv build-essential
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    # Opcional: binding Kinect v1
    python -m pip install -r requirements-kinect.txt

Execute com `python dance_interactive_poc.py`. Pressione q para sair.
Faixa inicial: --min-depth/--max-depth (mm). Atalhos Kinect: a/z diminuem/aumentam
Min Depth; s/x diminuem/aumentam Max Depth, em passos de 100 mm.
O modo Kinect usa profundidade em milimetros (DEPTH_MM).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

# Medido: o 1o frame do Kinect v1 chega em ~7 s neste equipamento.
KINECT_START_TIMEOUT = 20.0
# 0 significa "sem leitura" no mapa de profundidade do Kinect v1.
DEFAULT_MIN_DEPTH_MM = 800
DEFAULT_MAX_DEPTH_MM = 3000
DEPTH_STEP_MM = 100


class KinectCaptureShutdownError(RuntimeError):
    """Raised when a stuck native Kinect call makes fallback unsafe."""


class KinectV1Capturer:
    """Le RGB e profundidade em uma thread e publica somente o frame mais novo."""

    def __init__(
        self,
        min_depth: int = DEFAULT_MIN_DEPTH_MM,
        max_depth: int = DEFAULT_MAX_DEPTH_MM,
    ) -> None:
        import freenect

        self._freenect = freenect
        self._depth_format = getattr(freenect, "DEPTH_MM", None)
        if self._depth_format is None:
            raise RuntimeError("Esta instalacao do freenect nao oferece DEPTH_MM")

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._frame: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._frame_captured_at = 0.0
        self._last_read_timestamp = 0.0
        self._min_depth = min_depth
        self._max_depth = max_depth
        self.error: Optional[Exception] = None

        self._thread = threading.Thread(
            target=self._capture_loop, name="kinect-v1-capture", daemon=True
        )
        self._thread.start()
        print("Iniciando Kinect v1 (o primeiro frame pode levar ate ~15 s)...", flush=True)
        if not self._ready_event.wait(timeout=KINECT_START_TIMEOUT):
            self.close()
            if self._thread.is_alive():
                raise KinectCaptureShutdownError(
                    "A leitura nativa do Kinect continua bloqueada apos o pedido de parada; "
                    "reinicie o processo e verifique USB/alimentacao antes de tentar novamente"
                )
            raise RuntimeError("Timeout ao iniciar a captura do Kinect v1")
        if self.error is not None:
            error = self.error
            self.close()
            raise RuntimeError(f"Falha ao iniciar o Kinect v1: {error}") from error

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                depth_packet = self._freenect.sync_get_depth(format=self._depth_format)
                if depth_packet is None:
                    raise RuntimeError("O Kinect nao forneceu frame de profundidade")
                video_packet = self._freenect.sync_get_video(
                    format=self._freenect.VIDEO_RGB
                )
                if video_packet is None:
                    raise RuntimeError("O Kinect nao forneceu frame RGB")
                depth, _ = depth_packet
                rgb, _ = video_packet
                # O array do driver e liberado em sync_stop(); copiar evita acesso a memoria invalida.
                depth = np.array(depth, copy=True)
                rgb = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)

                with self._lock:
                    min_depth, max_depth = self._min_depth, self._max_depth
                body_mask = cv2.inRange(depth, min_depth, max_depth)
                with self._lock:
                    self._frame = (rgb, depth, body_mask)
                    self._frame_captured_at = time.perf_counter()
                self._ready_event.set()
        except Exception as exc:
            self.error = exc
            self._stop_event.set()
            self._ready_event.set()
        finally:
            # sync_stop() so e seguro na mesma thread que chama sync_get_*.
            if self._ready_event.is_set():
                try:
                    self._freenect.sync_stop()
                except Exception as exc:
                    print(f"Aviso ao parar o Kinect: {exc}", file=sys.stderr)

    def read(
        self,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Retorna imediatamente o RGB, profundidade em mm e mascara mais recentes."""
        with self._lock:
            if self._frame is None:
                return None
            self._last_read_timestamp = self._frame_captured_at
            return self._frame

    @property
    def last_read_timestamp(self) -> float:
        with self._lock:
            return self._last_read_timestamp

    def adjust_depth(self, min_delta: int = 0, max_delta: int = 0) -> None:
        with self._lock:
            self._min_depth = max(1, min(10000, self._min_depth + min_delta))
            self._max_depth = max(self._min_depth + 1, self._max_depth + max_delta)

    @property
    def depth_range(self) -> tuple[int, int]:
        with self._lock:
            return self._min_depth, self._max_depth

    @property
    def pose_landmarks(self):
        return None

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            print(
                "Aviso: a leitura nativa do Kinect nao respondeu ao pedido de parada; "
                "a thread daemon nao impedira o encerramento do processo.",
                file=sys.stderr,
            )


class WebcamCapturer:
    """Fallback de webcam com segmentacao e landmarks de pose MediaPipe."""

    def __init__(self, camera_index: int = 0) -> None:
        import mediapipe as mp

        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError("Nao foi possivel abrir a webcam")

        self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        self._pose = mp.solutions.pose.Pose(
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_pose = mp.solutions.pose
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._frame_captured_at = 0.0
        self._last_read_timestamp = 0.0
        self._pose_landmarks = None
        self.error: Optional[Exception] = None
        self._thread = threading.Thread(
            target=self._capture_loop, name="webcam-capture", daemon=False
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                ok, bgr = self._capture.read()
                if not ok:
                    raise RuntimeError("Falha ao ler frame da webcam")
                captured_at = time.perf_counter()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                segmentation = self._segmenter.process(rgb)
                pose = self._pose.process(rgb)
                mask = (segmentation.segmentation_mask > 0.4).astype(np.uint8) * 255
                depth = np.zeros(mask.shape, dtype=np.uint16)
                with self._lock:
                    self._frame = (bgr, depth, mask)
                    self._frame_captured_at = captured_at
                    self._pose_landmarks = pose.pose_landmarks
        except Exception as exc:
            self.error = exc
            self._stop_event.set()

    def read(
        self,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return None
            self._last_read_timestamp = self._frame_captured_at
            return self._frame

    @property
    def last_read_timestamp(self) -> float:
        with self._lock:
            return self._last_read_timestamp

    def adjust_depth(self, min_delta: int = 0, max_delta: int = 0) -> None:
        return

    @property
    def depth_range(self) -> tuple[int, int]:
        return 0, 0

    @property
    def pose_landmarks(self):
        with self._lock:
            return self._pose_landmarks

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=3.0)
        self._capture.release()
        self._segmenter.close()
        self._pose.close()
        if self._thread.is_alive():
            raise RuntimeError("A thread da webcam nao encerrou em 3 segundos")


def create_capturer(min_depth: int = DEFAULT_MIN_DEPTH_MM, max_depth: int = DEFAULT_MAX_DEPTH_MM):
    try:
        capturer = KinectV1Capturer(min_depth, max_depth)
        print("Kinect v1 detectado; usando profundidade em milimetros.")
        return capturer, "Kinect v1"
    except Exception as exc:
        print(f"Kinect v1 indisponivel ({exc}); tentando webcam + MediaPipe.")
        try:
            capturer = WebcamCapturer()
        except Exception as webcam_error:
            raise RuntimeError(
                f"Nenhum backend de captura disponivel. "
                f"Kinect: {exc}. Webcam: {webcam_error}."
            ) from webcam_error
        print("Webcam ativa com segmentacao e pose MediaPipe.")
        return capturer, "Webcam + MediaPipe"


def latency_color(latency_ms: float) -> tuple[int, int, int]:
    if latency_ms <= 35.0:
        return (0, 210, 0)
    if latency_ms <= 45.0:
        return (0, 220, 255)
    return (0, 0, 255)


def make_floor_visual(mask: np.ndarray, elapsed: float, phase: float) -> np.ndarray:
    height, width = mask.shape
    floor = np.zeros((height, width, 3), dtype=np.uint8)
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] > 0:
        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
        radius = int(30 + (elapsed % 1.5) / 1.5 * max(height, width) * 0.55)
        for ring in range(3):
            current_radius = max(1, radius - ring * 42)
            cv2.circle(floor, center, current_radius, (255, 150, 35), 2, cv2.LINE_AA)
        cv2.circle(floor, center, 8, (255, 240, 190), -1, cv2.LINE_AA)

    # Ondas horizontais sutis mantem o piso vivo mesmo quando o corpo esta parado.
    y = int((phase * 45) % max(height, 1))
    cv2.line(floor, (0, y), (width - 1, y), (24, 48, 52), 1, cv2.LINE_AA)
    return floor


def make_body_visual(mask: np.ndarray, elapsed: float) -> np.ndarray:
    height, width = mask.shape
    y, x = np.indices((height, width), dtype=np.float32)
    wave = (np.sin(x * 0.055 + elapsed * 3.0) + np.cos(y * 0.065 - elapsed * 2.0))
    glow = np.clip((wave + 2.0) * 58, 0, 255).astype(np.uint8)
    pattern = np.zeros((height, width, 3), dtype=np.uint8)
    pattern[:, :, 0] = glow
    pattern[:, :, 1] = np.clip(glow * 0.7 + 28, 0, 255).astype(np.uint8)
    pattern[:, :, 2] = np.clip(245 - glow * 0.45, 0, 255).astype(np.uint8)
    # bitwise_and garante preto absoluto fora da mascara binaria.
    return cv2.bitwise_and(pattern, pattern, mask=mask)


def make_composite_preview(floor: np.ndarray, body: np.ndarray) -> np.ndarray:
    # Soma saturada: dois projetores sobrepostos somam luz, nao se substituem.
    return cv2.add(floor, body)


def draw_debug_overlay(
    debug: np.ndarray,
    source: str,
    fps: float,
    latency_ms: float,
    depth_range: tuple[int, int],
) -> None:
    color = latency_color(latency_ms)
    lines = [
        f"FPS: {fps:.1f}",
        f"Latencia: {latency_ms:.1f} ms",
        f"Min Depth: {depth_range[0]} mm",
        f"Max Depth: {depth_range[1]} mm",
        source,
    ]
    cv2.rectangle(debug, (8, 8), (245, 122), (0, 0, 0), -1)
    for index, text in enumerate(lines):
        cv2.putText(
            debug,
            text,
            (16, 30 + index * 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color if index == 1 else (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PoC de danca interativa com Kinect v1 ou webcam.")
    parser.add_argument("--min-depth", type=int, default=DEFAULT_MIN_DEPTH_MM, help="profundidade minima em mm (padrao: %(default)s)")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH_MM, help="profundidade maxima em mm (padrao: %(default)s)")
    args = parser.parse_args(argv)
    if not 1 <= args.min_depth < args.max_depth:
        parser.error("--min-depth deve ser >= 1 e menor que --max-depth")
    return args


def main() -> int:
    args = parse_args()
    try:
        capturer, source = create_capturer(args.min_depth, args.max_depth)
    except RuntimeError as exc:
        print(f"Erro ao iniciar captura: {exc}", file=sys.stderr)
        return 1
    windows = (
        "Debug & Tracking",
        "Projetor 1 - Chao/Fundo",
        "Projetor 2 - Corpo/Frontal",
        "Simulacao - Chao + Corpo",
    )
    cv2.namedWindow(windows[0], cv2.WINDOW_NORMAL)
    cv2.namedWindow(windows[1], cv2.WINDOW_NORMAL)
    cv2.namedWindow(windows[2], cv2.WINDOW_NORMAL)
    cv2.namedWindow(windows[3], cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(windows[1], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(windows[2], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    recent_latency = deque(maxlen=30)
    recent_frame_times = deque(maxlen=30)
    last_latency_ms = 0.0
    phase = 0.0
    start_time = time.perf_counter()
    try:
        while True:
            if capturer.error is not None:
                raise RuntimeError(f"Falha no backend de captura: {capturer.error}")
            packet = capturer.read()
            if packet is None:
                cv2.waitKey(1)
                continue

            rgb, depth, body_mask = packet
            frame_start = capturer.last_read_timestamp
            elapsed = frame_start - start_time
            recent_frame_times.append(time.perf_counter())
            debug = rgb.copy()
            tinted = np.zeros_like(debug)
            tinted[:, :, 1] = 200
            overlay = cv2.bitwise_and(tinted, tinted, mask=body_mask)
            debug = cv2.addWeighted(debug, 1.0, overlay, 0.28, 0.0)

            landmarks = capturer.pose_landmarks
            if landmarks is not None:
                capturer._mp_drawing.draw_landmarks(
                    debug, landmarks, capturer._mp_pose.POSE_CONNECTIONS
                )
            else:
                contours, _ = cv2.findContours(
                    body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(debug, contours, -1, (40, 230, 255), 1)

            fps = (
                (len(recent_frame_times) - 1)
                / (recent_frame_times[-1] - recent_frame_times[0])
                if len(recent_frame_times) > 1
                else 0.0
            )
            depth_range = capturer.depth_range
            draw_debug_overlay(debug, source, fps, last_latency_ms, depth_range)
            floor = make_floor_visual(body_mask, elapsed, phase)
            body = make_body_visual(body_mask, elapsed)
            composite = make_composite_preview(floor, body)

            cv2.imshow(windows[0], debug)
            cv2.imshow(windows[1], floor)
            cv2.imshow(windows[2], body)
            cv2.imshow(windows[3], composite)
            key = cv2.waitKey(1) & 0xFF
            last_latency_ms = (time.perf_counter() - frame_start) * 1000.0
            recent_latency.append(last_latency_ms)
            phase += last_latency_ms / 1000.0
            if key == ord("q"):
                break
            if source == "Kinect v1":
                if key == ord("a"):
                    capturer.adjust_depth(min_delta=-DEPTH_STEP_MM)
                elif key == ord("z"):
                    capturer.adjust_depth(min_delta=DEPTH_STEP_MM)
                elif key == ord("s"):
                    capturer.adjust_depth(max_delta=-DEPTH_STEP_MM)
                elif key == ord("x"):
                    capturer.adjust_depth(max_delta=DEPTH_STEP_MM)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario; encerrando captura...", file=sys.stderr)
    finally:
        capturer.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
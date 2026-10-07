#!/usr/bin/env python3
"""PoC de danca interativa com Kinect v1, webcam USB ou arquivo de video.

Instalacao no Ubuntu (para Kinect, instale primeiro as libs nativas):
    sudo apt-get install libfreenect-dev freenect python3-dev python3-venv build-essential
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    # Opcional: binding Kinect v1
    python -m pip install -r requirements-kinect.txt

Execute com `python dance_interactive_poc.py`. Pressione q para sair.
Fonte: --source kinect|webcam|arquivo.mp4; pressione m para alternar as fontes configuradas.
Use --fallback-video para reserva quando Kinect e webcam falharem.
Algoritmo: --mode kinect|mog2|optical_flow|yolo. Fontes 2D usam MOG2 por padrao.
Faixa inicial: --min-depth/--max-depth (mm). Atalhos Kinect: a/z diminuem/aumentam
Min Depth; s/x diminuem/aumentam Max Depth, em passos de 100 mm. Pressione b
com a sala vazia para capturar o fundo e recortar apenas o que se move a frente.
O modo Kinect usa profundidade em milimetros (DEPTH_MM).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Protocol

import cv2
import numpy as np

# Medido: o 1o frame do Kinect v1 chega em ~7 s neste equipamento.
KINECT_START_TIMEOUT = 20.0
# 0 significa "sem leitura" no mapa de profundidade do Kinect v1.
DEFAULT_MIN_DEPTH_MM = 800
DEFAULT_MAX_DEPTH_MM = 3000
DEPTH_STEP_MM = 100
BACKGROUND_SAMPLE_COUNT = 15
BACKGROUND_DIFF_MM = 80
FrameData = tuple[np.ndarray, np.ndarray, np.ndarray]
CaptureResult = tuple[
    bool,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]


class CaptureSource(Protocol):
    error: Optional[Exception]

    @property
    def source_key(self) -> str: ...

    def read(self) -> CaptureResult: ...

    @property
    def source_label(self) -> str: ...

    @property
    def last_read_timestamp(self) -> float: ...

    @property
    def depth_range(self) -> tuple[int, int]: ...

    @property
    def background_active(self) -> bool: ...

    def adjust_depth(self, min_delta: int = 0, max_delta: int = 0) -> None: ...

    def capture_background(self) -> None: ...

    def close(self) -> None: ...


def make_kinect_body_mask(
    depth: np.ndarray,
    min_depth: int,
    max_depth: int,
    background_depth: Optional[np.ndarray] = None,
) -> np.ndarray:
    mask = cv2.inRange(depth, min_depth, max_depth)
    if background_depth is not None:
        closer = (
            (depth > 0)
            & (background_depth > 0)
            & (background_depth.astype(np.int32) - depth.astype(np.int32) >= BACKGROUND_DIFF_MM)
        )
        mask = cv2.bitwise_and(mask, closer.astype(np.uint8) * 255)
    return mask


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
        self._frame: Optional[FrameData] = None
        self._frame_captured_at = 0.0
        self._last_read_timestamp = 0.0
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._background_depth: Optional[np.ndarray] = None
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
                rgb = np.array(rgb, copy=True)

                with self._lock:
                    min_depth, max_depth = self._min_depth, self._max_depth
                    background_depth = self._background_depth
                body_mask = make_kinect_body_mask(
                    depth, min_depth, max_depth, background_depth
                )
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
    ) -> CaptureResult:
        with self._lock:
            if self._frame is None:
                return False, None, None, None
            self._last_read_timestamp = self._frame_captured_at
            return (True, *self._frame)

    @property
    def source_label(self) -> str:
        min_depth, max_depth = self.depth_range
        return f"KINECT v1 (Depth RAW: {min_depth}-{max_depth} mm)"

    @property
    def source_key(self) -> str:
        return "kinect"

    @property
    def last_read_timestamp(self) -> float:
        with self._lock:
            return self._last_read_timestamp

    def adjust_depth(self, min_delta: int = 0, max_delta: int = 0) -> None:
        with self._lock:
            self._min_depth = max(1, min(10000, self._min_depth + min_delta))
            self._max_depth = max(self._min_depth + 1, self._max_depth + max_delta)

    def capture_background(self) -> None:
        samples = []
        last_timestamp = -1.0
        deadline = time.perf_counter() + 5.0
        while len(samples) < BACKGROUND_SAMPLE_COUNT and time.perf_counter() < deadline:
            with self._lock:
                timestamp = self._frame_captured_at
                if self._frame is not None and timestamp != last_timestamp:
                    samples.append(self._frame[1].copy())
                    last_timestamp = timestamp
            time.sleep(0.002)
        if len(samples) < BACKGROUND_SAMPLE_COUNT:
            raise RuntimeError("Nao foi possivel capturar frames suficientes do Kinect")

        background = np.ma.median(
            np.ma.masked_equal(np.stack(samples), 0), axis=0
        ).filled(0).astype(np.uint16)
        with self._lock:
            self._background_depth = background

    @property
    def background_active(self) -> bool:
        with self._lock:
            return self._background_depth is not None

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
    """Webcam USB frame source."""

    def __init__(self, camera_index: int = 0) -> None:
        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError("Nao foi possivel abrir a webcam")

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame: Optional[FrameData] = None
        self._frame_captured_at = 0.0
        self._last_read_timestamp = 0.0
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
                mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
                depth = np.zeros(mask.shape, dtype=np.uint16)
                with self._lock:
                    self._frame = (rgb, depth, mask)
                    self._frame_captured_at = captured_at
        except Exception as exc:
            self.error = exc
            self._stop_event.set()

    def read(
        self,
    ) -> CaptureResult:
        with self._lock:
            if self._frame is None:
                return False, None, None, None
            self._last_read_timestamp = self._frame_captured_at
            return (True, *self._frame)

    @property
    def source_label(self) -> str:
        return "WEBCAM USB"

    @property
    def source_key(self) -> str:
        return "webcam"

    @property
    def last_read_timestamp(self) -> float:
        with self._lock:
            return self._last_read_timestamp

    def adjust_depth(self, min_delta: int = 0, max_delta: int = 0) -> None:
        return

    def capture_background(self) -> None:
        raise RuntimeError("A calibracao de fundo requer o Kinect v1")

    @property
    def background_active(self) -> bool:
        return False

    @property
    def depth_range(self) -> tuple[int, int]:
        return 0, 0

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=3.0)
        self._capture.release()
        if self._thread.is_alive():
            raise RuntimeError("A thread da webcam nao encerrou em 3 segundos")


class VideoCapturer:
    """Video file source that loops at its nominal frame rate."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        self._capture = cv2.VideoCapture(self.path)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Nao foi possivel abrir o video: {self.path}")

        fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._frame_period = 1.0 / (fps if 1.0 <= fps <= 120.0 else 30.0)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame: Optional[FrameData] = None
        self._frame_captured_at = 0.0
        self._last_read_timestamp = 0.0
        self.error: Optional[Exception] = None
        self._thread = threading.Thread(
            target=self._capture_loop, name="video-capture", daemon=False
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        next_frame_at = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                ok, bgr = self._capture.read()
                if not ok:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, bgr = self._capture.read()
                    if not ok:
                        raise RuntimeError(f"O arquivo de video esta vazio: {self.path}")

                captured_at = time.perf_counter()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
                depth = np.zeros(mask.shape, dtype=np.uint16)
                with self._lock:
                    self._frame = (rgb, depth, mask)
                    self._frame_captured_at = captured_at

                next_frame_at += self._frame_period
                if next_frame_at < time.perf_counter():
                    next_frame_at = time.perf_counter()
                self._stop_event.wait(max(0.0, next_frame_at - time.perf_counter()))
        except Exception as exc:
            self.error = exc
            self._stop_event.set()

    def read(self) -> CaptureResult:
        with self._lock:
            if self._frame is None:
                return False, None, None, None
            self._last_read_timestamp = self._frame_captured_at
            return (True, *self._frame)

    @property
    def source_label(self) -> str:
        return f"VIDEO [{os.path.basename(self.path)}]"

    @property
    def source_key(self) -> str:
        return self.path

    @property
    def last_read_timestamp(self) -> float:
        with self._lock:
            return self._last_read_timestamp

    @property
    def depth_range(self) -> tuple[int, int]:
        return 0, 0

    @property
    def background_active(self) -> bool:
        return False

    def adjust_depth(self, min_delta: int = 0, max_delta: int = 0) -> None:
        return

    def capture_background(self) -> None:
        raise RuntimeError("A calibracao de fundo requer o Kinect v1")

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=3.0)
        self._capture.release()
        if self._thread.is_alive():
            raise RuntimeError("A thread de video nao encerrou em 3 segundos")


def create_capturer(
    source: str = "kinect",
    min_depth: int = DEFAULT_MIN_DEPTH_MM,
    max_depth: int = DEFAULT_MAX_DEPTH_MM,
    fallback_video: Optional[str] = None,
) -> CaptureSource:
    if source == "kinect":
        try:
            capturer = KinectV1Capturer(min_depth, max_depth)
            print("Kinect v1 detectado; usando profundidade em milimetros.")
            return capturer
        except Exception as exc:
            print(
                f"Aviso: Kinect indisponivel ({exc}); tentando webcam USB + MOG2.",
                file=sys.stderr,
            )
            try:
                return WebcamCapturer()
            except Exception as webcam_error:
                if fallback_video:
                    print("Webcam indisponivel; tentando video de fallback.", file=sys.stderr)
                    return VideoCapturer(fallback_video)
                raise RuntimeError(
                    f"Kinect indisponivel: {exc}. Webcam indisponivel: {webcam_error}. "
                    "Use --fallback-video caminho.mp4 para uma fonte de reserva."
                ) from webcam_error

    if source == "webcam":
        return WebcamCapturer()
    if os.path.splitext(source)[1].lower() not in {".mp4", ".mov", ".m4v"}:
        raise ValueError("--source deve ser 'kinect', 'webcam' ou um arquivo MP4/MOV")
    return VideoCapturer(source)


@dataclass(slots=True)
class VisionResult:
    body_mask: np.ndarray
    debug_frame: np.ndarray
    body_frame: np.ndarray


class VisionProcessor:
    """Runs one selected 2D algorithm on each new RGB frame."""

    MODE_LABELS = {
        "mog2": "MOG2 (Subtracao de Fundo)",
        "optical_flow": "Optical Flow (Farneback)",
        "yolo": "YOLOv8n-pose",
    }
    SKELETON_EDGES = (
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    )
    MOG2_SCALE = 0.25
    FLOW_SCALE = 0.125
    FLOW_THRESHOLD = 0.15

    def __init__(self, mode: str) -> None:
        if mode not in self.MODE_LABELS:
            raise ValueError(f"Modo 2D desconhecido: {mode}")
        self.mode = mode
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._mog2 = None
        self._previous_gray: Optional[np.ndarray] = None
        self._yolo = None
        if mode == "mog2":
            self._reset_mog2()
        elif mode == "yolo":
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "O modo YOLO requer ultralytics; instale com "
                    "python -m pip install -r requirements.txt"
                ) from exc
            self._yolo = YOLO("yolov8n-pose.pt")

    @property
    def label(self) -> str:
        return self.MODE_LABELS[self.mode]

    def _reset_mog2(self) -> None:
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )

    def reset(self) -> None:
        if self.mode == "mog2":
            self._reset_mog2()
        elif self.mode == "optical_flow":
            self._previous_gray = None

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask

    @staticmethod
    def _empty_result(frame_bgr: np.ndarray) -> VisionResult:
        height, width = frame_bgr.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        return VisionResult(mask, frame_bgr.copy(), np.zeros_like(frame_bgr))

    def process(self, frame_rgb: np.ndarray) -> VisionResult:
        if self.mode == "mog2":
            height, width = frame_rgb.shape[:2]
            size = (
                max(1, int(width * self.MOG2_SCALE)),
                max(1, int(height * self.MOG2_SCALE)),
            )
            small_rgb = cv2.resize(frame_rgb, size, interpolation=cv2.INTER_AREA)
            small_mask = self._clean_mask(self._mog2.apply(small_rgb))
            mask = cv2.resize(
                small_mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
            debug = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(debug, contours, -1, (0, 255, 0), 1)
            return VisionResult(mask, debug, make_body_visual(mask))

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if self.mode == "optical_flow":
            small_gray = cv2.resize(
                gray,
                (
                    max(1, int(gray.shape[1] * self.FLOW_SCALE)),
                    max(1, int(gray.shape[0] * self.FLOW_SCALE)),
                ),
                interpolation=cv2.INTER_AREA,
            )
            result = self._process_flow(frame_rgb, small_gray)
            self._previous_gray = small_gray
            return result

        return self._process_yolo(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def _process_flow(self, frame_rgb: np.ndarray, gray: np.ndarray) -> VisionResult:
        height, width = frame_rgb.shape[:2]
        if self._previous_gray is None:
            return self._empty_result(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        flow = cv2.calcOpticalFlowFarneback(
            self._previous_gray, gray, None, 0.5, 1, 5, 1, 5, 1.1, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        moving = (magnitude > self.FLOW_THRESHOLD).astype(np.uint8) * 255
        small_mask = self._clean_mask(moving)
        mask = cv2.resize(
            small_mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
        debug = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        body = np.zeros_like(debug)
        scale = 1.0 / self.FLOW_SCALE
        flow_height, flow_width = gray.shape
        for y in range(3, flow_height, 4):
            for x in range(3, flow_width, 4):
                if magnitude[y, x] <= self.FLOW_THRESHOLD:
                    continue
                dx, dy = flow[y, x]
                start = (int(x * scale), int(y * scale))
                end = (
                    int((x + dx * 3) * scale),
                    int((y + dy * 3) * scale),
                )
                cv2.arrowedLine(debug, start, end, (0, 255, 255), 1, cv2.LINE_AA)
                if small_mask[y, x]:
                    cv2.arrowedLine(body, start, end, (0, 255, 255), 1, cv2.LINE_AA)
            body = cv2.bitwise_and(body, body, mask=mask)
        return VisionResult(mask, debug, body)

    @staticmethod
    def _to_numpy(value) -> Optional[np.ndarray]:
        if value is None:
            return None
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def _process_yolo(self, frame_bgr: np.ndarray) -> VisionResult:
        predictions = self._yolo(frame_bgr, verbose=False, imgsz=320)
        result = predictions[0]
        height, width = frame_bgr.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        debug = frame_bgr.copy()
        keypoints = getattr(result, "keypoints", None)
        points_by_person = self._to_numpy(getattr(keypoints, "xy", None))
        confidence = self._to_numpy(getattr(keypoints, "conf", None))
        boxes = self._to_numpy(getattr(getattr(result, "boxes", None), "xyxy", None))
        body = np.zeros_like(frame_bgr)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = np.rint(box[:4]).astype(int)
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 200, 255), 2)
        if points_by_person is not None:
            for person_index, points in enumerate(points_by_person):
                valid = (
                    np.isfinite(points).all(axis=1)
                    & (points[:, 0] >= 0)
                    & (points[:, 0] < width)
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < height)
                )
                if confidence is not None and person_index < len(confidence):
                    valid &= confidence[person_index] >= 0.25
                integer_points = np.rint(
                    np.nan_to_num(points, nan=-1.0, posinf=-1.0, neginf=-1.0)
                ).astype(int)
                for start, end in self.SKELETON_EDGES:
                    if valid[start] and valid[end]:
                        p1 = tuple(integer_points[start])
                        p2 = tuple(integer_points[end])
                        cv2.line(mask, p1, p2, 255, 24, cv2.LINE_AA)
                        cv2.line(debug, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.line(body, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)
                for point_index in range(5, len(integer_points)):
                    point = integer_points[point_index]
                    if valid[point_index]:
                        p = tuple(point)
                        cv2.circle(mask, p, 12, 255, -1, cv2.LINE_AA)
                        cv2.circle(debug, p, 4, (0, 255, 0), -1, cv2.LINE_AA)
                        cv2.circle(body, p, 4, (0, 255, 0), -1, cv2.LINE_AA)

                visible_head = integer_points[:5][valid[:5]]
                if len(visible_head):
                    head_center_array = np.rint(visible_head.mean(axis=0)).astype(int)
                    head_center = tuple(head_center_array)
                    radius = max(
                        8,
                        int(
                            np.linalg.norm(visible_head - head_center_array, axis=1).max()
                            + 5
                        ),
                    )
                    cv2.circle(mask, head_center, radius, 255, 1, cv2.LINE_AA)
                    cv2.circle(debug, head_center, radius, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.circle(body, head_center, radius, (0, 0, 255), 1, cv2.LINE_AA)

        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        return VisionResult(mask, debug, cv2.bitwise_and(body, body, mask=mask))


def latency_color(latency_ms: float) -> tuple[int, int, int]:
    if latency_ms <= 35.0:
        return (0, 210, 0)
    if latency_ms <= 45.0:
        return (0, 220, 255)
    return (0, 0, 255)


def make_body_visual(mask: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(mask)
    return cv2.merge((zeros, zeros, mask))


def draw_debug_overlay(
    debug: np.ndarray,
    source: str,
    mode: str,
    fps: float,
    latency_ms: float,
    depth_range: tuple[int, int],
    background_active: bool,
) -> None:
    color = latency_color(latency_ms)
    min_depth, max_depth = depth_range
    lines = [
        f"FPS: {fps:.1f}",
        f"Latencia: {latency_ms:.1f} ms",
        f"Modo Ativo: {mode}",
        f"Min Depth: {min_depth} mm" if max_depth else "Min Depth: N/A",
        f"Max Depth: {max_depth} mm" if max_depth else "Max Depth: N/A",
        (
            "Mascara: desativada"
            if not max_depth
            else "Fundo: calibrado"
            if background_active
            else "Fundo: pressione B vazio"
        ),
        f"Fonte Ativa: {source}",
    ]
    cv2.rectangle(debug, (8, 8), (630, 174), (0, 0, 0), -1)
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
    parser = argparse.ArgumentParser(description="PoC de danca interativa com captura e processamento 2D intercambiaveis.")
    parser.add_argument(
        "--source",
        default="kinect",
        metavar="FONTE",
        help="kinect, webcam ou caminho MP4/MOV (padrao: kinect; tecla m alterna fontes)",
    )
    parser.add_argument(
        "--fallback-video",
        metavar="ARQUIVO",
        help="video MP4/MOV opcional se Kinect e webcam estiverem indisponiveis",
    )
    parser.add_argument(
        "--mode",
        choices=("kinect", "mog2", "optical_flow", "yolo"),
        default=None,
        help="algoritmo 2D: kinect, mog2, optical_flow ou yolo (padrao: kinect para Kinect, mog2 para outras fontes)",
    )
    parser.add_argument("--min-depth", type=int, default=DEFAULT_MIN_DEPTH_MM, help="profundidade minima em mm (padrao: %(default)s)")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH_MM, help="profundidade maxima em mm (padrao: %(default)s)")
    args = parser.parse_args(argv)
    if not 1 <= args.min_depth < args.max_depth:
        parser.error("--min-depth deve ser >= 1 e menor que --max-depth")
    if args.source not in {"kinect", "webcam"} and os.path.splitext(args.source)[1].lower() not in {".mp4", ".mov", ".m4v"}:
        parser.error("--source deve ser 'kinect', 'webcam' ou um arquivo MP4/MOV")
    if args.fallback_video and os.path.splitext(args.fallback_video)[1].lower() not in {".mp4", ".mov", ".m4v"}:
        parser.error("--fallback-video deve apontar para um arquivo MP4/MOV")
    if args.mode is None:
        args.mode = "kinect" if args.source == "kinect" else "mog2"
    if args.mode == "kinect" and args.source != "kinect":
        parser.error("--mode kinect requer --source kinect")
    return args


def resolve_active_mode(requested_mode: str, capturer: CaptureSource) -> str:
    if requested_mode == "kinect" and not isinstance(capturer, KinectV1Capturer):
        print("Kinect indisponivel; usando MOG2 no modo de fallback.", file=sys.stderr)
        return "mog2"
    return requested_mode


def make_vision_processor(mode: str) -> Optional[VisionProcessor]:
    return None if mode == "kinect" else VisionProcessor(mode)


def main() -> int:
    args = parse_args()
    try:
        capturer = create_capturer(
            args.source, args.min_depth, args.max_depth, args.fallback_video
        )
    except RuntimeError as exc:
        print(f"Erro ao iniciar captura: {exc}", file=sys.stderr)
        return 1
    active_mode = resolve_active_mode(args.mode, capturer)
    try:
        processor = make_vision_processor(active_mode)
    except Exception as exc:
        capturer.close()
        print(f"Erro ao iniciar modo {active_mode}: {exc}", file=sys.stderr)
        return 1
    video_choice = args.source if args.source not in {"kinect", "webcam"} else args.fallback_video
    source_choices = ["kinect", "webcam"]
    if video_choice:
        video_choice = os.path.abspath(video_choice)
        if video_choice not in source_choices:
            source_choices.append(video_choice)
    windows = (
        "Debug & Tracking",
        "Projetor 2 - Corpo/Frontal",
    )
    cv2.namedWindow(windows[0], cv2.WINDOW_NORMAL)
    cv2.namedWindow(windows[1], cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(windows[1], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    recent_frame_times = deque(maxlen=30)
    last_latency_ms = 0.0
    last_processed_timestamp: Optional[float] = None
    cached_images: Optional[tuple[np.ndarray, np.ndarray]] = None
    try:
        while True:
            if capturer.error is not None:
                raise RuntimeError(f"Falha no backend de captura: {capturer.error}")
            success, frame_rgb, depth, body_mask = capturer.read()
            if not success or frame_rgb is None or depth is None or body_mask is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            frame_start = capturer.last_read_timestamp
            is_new_frame = frame_start != last_processed_timestamp
            if is_new_frame:
                recent_frame_times.append(frame_start)
                if active_mode == "kinect":
                    debug = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    tinted = np.zeros_like(debug)
                    tinted[:, :, 1] = 200
                    overlay = cv2.bitwise_and(tinted, tinted, mask=body_mask)
                    debug = cv2.addWeighted(debug, 1.0, overlay, 0.28, 0.0)
                    contours, _ = cv2.findContours(
                        body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(debug, contours, -1, (40, 230, 255), 1)
                    vision_result = None
                else:
                    vision_result = processor.process(frame_rgb)
                    body_mask = vision_result.body_mask
                    debug = vision_result.debug_frame

                fps = (
                    (len(recent_frame_times) - 1)
                    / (recent_frame_times[-1] - recent_frame_times[0])
                    if len(recent_frame_times) > 1
                    else 0.0
                )
                depth_range = capturer.depth_range if active_mode == "kinect" else (0, 0)
                mode_label = (
                    f"KINECT v1 (Depth RAW: {depth_range[0]}-{depth_range[1]} mm)"
                    if active_mode == "kinect"
                    else processor.label
                )
                draw_debug_overlay(
                    debug,
                    capturer.source_label,
                    mode_label,
                    fps,
                    last_latency_ms,
                    depth_range,
                    capturer.background_active if active_mode == "kinect" else False,
                )
                body = (
                    vision_result.body_frame
                    if vision_result is not None
                    else make_body_visual(body_mask)
                )
                cached_images = (debug, body)
                last_processed_timestamp = frame_start

            if cached_images is None:
                cv2.waitKey(1)
                continue
            for window, image in zip(windows, cached_images):
                cv2.imshow(window, image)
            key = cv2.waitKey(1) & 0xFF
            if is_new_frame:
                last_latency_ms = (time.perf_counter() - frame_start) * 1000.0
            if key == ord("q"):
                break
            if key == ord("m"):
                previous = capturer
                previous_key = previous.source_key
                next_index = (source_choices.index(previous_key) + 1) % len(source_choices)
                next_key = source_choices[next_index]
                print(f"Alternando fonte: {previous.source_label} -> {next_key}")

                if next_key == "kinect":
                    previous.close()
                    capturer = None
                    try:
                        capturer = create_capturer(
                            next_key,
                            args.min_depth,
                            args.max_depth,
                            args.fallback_video,
                        )
                    except Exception as exc:
                        print(f"Falha ao trocar para Kinect: {exc}", file=sys.stderr)
                        try:
                            capturer = create_capturer(
                                previous_key, args.min_depth, args.max_depth
                            )
                        except Exception as restore_error:
                            capturer = None
                            print(
                                f"Nao foi possivel restaurar {previous_key}: {restore_error}",
                                file=sys.stderr,
                            )
                            break
                else:
                    try:
                        replacement = create_capturer(
                            next_key, args.min_depth, args.max_depth
                        )
                    except Exception as exc:
                        print(f"Falha ao trocar para {next_key}: {exc}", file=sys.stderr)
                        continue
                    previous.close()
                    capturer = replacement

                if capturer is not None:
                    previous_mode = active_mode
                    active_mode = resolve_active_mode(args.mode, capturer)
                    if active_mode != previous_mode:
                        processor = make_vision_processor(active_mode)
                    elif processor is not None:
                        processor.reset()
                    recent_frame_times.clear()
                    last_latency_ms = 0.0
                    last_processed_timestamp = None
                    cached_images = None
                continue
            if capturer is None:
                break
            if isinstance(capturer, KinectV1Capturer) and active_mode == "kinect":
                if key == ord("a"):
                    capturer.adjust_depth(min_delta=-DEPTH_STEP_MM)
                elif key == ord("z"):
                    capturer.adjust_depth(min_delta=DEPTH_STEP_MM)
                elif key == ord("s"):
                    capturer.adjust_depth(max_delta=-DEPTH_STEP_MM)
                elif key == ord("x"):
                    capturer.adjust_depth(max_delta=DEPTH_STEP_MM)
                elif key == ord("b"):
                    print("Capture o fundo com a sala vazia; amostrando 15 frames...")
                    try:
                        capturer.capture_background()
                    except RuntimeError as exc:
                        print(f"Falha ao calibrar fundo: {exc}", file=sys.stderr)
                    else:
                        print("Fundo calibrado; apenas objetos mais proximos entrarao na mascara.")
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario; encerrando captura...", file=sys.stderr)
    finally:
        if capturer is not None:
            capturer.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
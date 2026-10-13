#!/usr/bin/env python3
"""Mede o custo dos modos do --pi sem camera, para comparar perfis e maquinas.

Gera quadros 640x480 como os do Kinect v1 (RGB + mascara de profundidade) com
uma figura que anda pela cena e roda o VisionProcessor (xadrez ou paisagens).
So precisa de numpy e opencv; roda igual num PC e num Raspberry Pi.

    python pi_benchmark.py                 # perfis normal e --pi, regular e mix
    python pi_benchmark.py --display       # inclui o imshow em tela cheia
    python pi_benchmark.py --profile pi --layout regular --silhouette on
    python pi_benchmark.py --mode paisagens   # backends numpy e opencv do borrao

O relogio dos pulsos avanca 1/30 s por quadro, como numa captura a 30 fps, sem
esperar de verdade: pausas reais baixam o clock da CPU e distorcem a medida.
"""

from __future__ import annotations

import argparse
import itertools
import time

import cv2
import numpy as np

import dance_interactive_poc as poc

FRAME_SIZE = (640, 480)
CAPTURE_FPS = 30.0
PROFILES = {"normal": poc.PlaidRender(), "pi": poc.PI_PLAID_RENDER}


def synthetic_frames(count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Quadros (BGR, mascara) com um corpo simplificado andando e mexendo o braco."""
    width, height = FRAME_SIZE
    rng = np.random.default_rng(1)
    background = cv2.GaussianBlur(
        rng.integers(30, 90, (height, width, 3), dtype=np.uint8), (0, 0), 3
    )
    frames = []
    for index in range(count):
        mask = np.zeros((height, width), np.uint8)
        x = 80 + (index % 60) * 8
        arm_y = 270 + int(40 * np.sin(index / 5))
        cv2.ellipse(mask, (x, 120), (28, 34), 0, 0, 360, 255, -1)
        cv2.rectangle(mask, (x - 38, 160), (x + 38, 360), 255, -1)
        cv2.line(mask, (x - 38, 175), (x - 95, arm_y), 255, 18)
        cv2.line(mask, (x - 20, 360), (x - 45, height - 1), 255, 22)
        cv2.line(mask, (x + 20, 360), (x + 50, height - 1), 255, 22)
        frame = background.copy()
        frame[mask > 0] = (170, 150, 140)
        frames.append((frame, mask))
    return frames


def run(
    frames: list[tuple[np.ndarray, np.ndarray]],
    profile: str,
    layout: str,
    silhouette: bool,
    display: bool,
) -> dict:
    processor = poc.VisionProcessor(
        "xadrez",
        plaid_layout=layout,
        silhouette=silhouette,
        render=PROFILES[profile],
    )
    # Mesmo sorteio em todas as combinacoes, para comparar so o perfil.
    processor._rng = np.random.default_rng(0)
    processor._plaid_patterns = [
        (poc.make_madras_sett(processor._rng), (0.3, 0.6), angle)
        for angle in poc.make_plaid_angle_offsets(processor._rng, poc.PLAID_PATTERN_COUNT)
    ]
    simulated = itertools.count()
    processor.clock = lambda: next(simulated) / CAPTURE_FPS
    window = "pi_benchmark"
    if display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    process_ms, display_ms = [], []
    for frame, mask in frames:
        start = time.perf_counter()
        result = processor.process(frame, mask)
        process_ms.append((time.perf_counter() - start) * 1000.0)
        if display:
            start = time.perf_counter()
            cv2.imshow(window, result.body_frame)
            cv2.waitKey(1)
            display_ms.append((time.perf_counter() - start) * 1000.0)
    if display:
        cv2.destroyWindow(window)
    # Descarta o aquecimento: os pulsos levam alguns segundos para encher a tela.
    warmup = min(len(frames) // 3, int(3 * CAPTURE_FPS))
    process = np.asarray(process_ms[warmup:])
    total = process + (np.asarray(display_ms[warmup:]) if display else 0.0)
    return {
        "lines": len(processor._field),
        "pulses": processor._field.pulse_count,
        "process": float(np.median(process)),
        "process_p95": float(np.percentile(process, 95)),
        "display": float(np.median(display_ms[warmup:])) if display else None,
        "fps": 1000.0 / float(np.median(total)),
    }


def run_landscape(
    frames: list[tuple[np.ndarray, np.ndarray]], backend: str, display: bool
) -> dict:
    processor = poc.VisionProcessor("paisagens", render=poc.PI_PLAID_RENDER)
    processor.process(frames[0][0])
    processor._field.backend = backend
    window = "pi_benchmark"
    if display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    process_ms, display_ms = [], []
    for frame, _ in frames[1:]:
        start = time.perf_counter()
        result = processor.process(frame)
        process_ms.append((time.perf_counter() - start) * 1000.0)
        if display:
            start = time.perf_counter()
            cv2.imshow(window, result.body_frame)
            cv2.waitKey(1)
            display_ms.append((time.perf_counter() - start) * 1000.0)
    if display:
        cv2.destroyWindow(window)
    process = np.asarray(process_ms)
    total = process + (np.asarray(display_ms) if display else 0.0)
    return {
        "process": float(np.median(process)),
        "process_p95": float(np.percentile(process, 95)),
        "display": float(np.median(display_ms)) if display else None,
        "fps": 1000.0 / float(np.median(total)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("xadrez", "paisagens"), default="xadrez")
    parser.add_argument("--profile", choices=(*PROFILES, "all"), default="all")
    parser.add_argument("--layout", choices=(*poc.PLAID_LAYOUTS, "all"), default="all")
    parser.add_argument("--silhouette", choices=("on", "off", "all"), default="all")
    parser.add_argument("--frames", type=int, default=300, help="quadros por combinacao")
    parser.add_argument(
        "--display", action="store_true", help="mede tambem o imshow em tela cheia"
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="limita as threads do OpenCV"
    )
    args = parser.parse_args()
    if args.threads is not None:
        cv2.setNumThreads(args.threads)

    frames = synthetic_frames(args.frames)
    if args.mode == "paisagens":
        print(
            f"paisagens, {FRAME_SIZE[0]}x{FRAME_SIZE[1]}, {args.frames} quadros, "
            f"OpenCV {cv2.__version__}, {cv2.getNumThreads()} threads"
        )
        print(f"{'mapas':7s} {'proc. ms':>9s} {'p95':>6s}" + (f" {'tela ms':>8s}" if args.display else "") + f" {'FPS max':>8s}")
        for backend in ("numpy", "opencv"):
            result = run_landscape(frames, backend, args.display)
            row = f"{backend:7s} {result['process']:9.1f} {result['process_p95']:6.1f}"
            if args.display:
                row += f" {result['display']:8.1f}"
            print(row + f" {result['fps']:8.1f}")
        return 0
    profiles = list(PROFILES) if args.profile == "all" else [args.profile]
    layouts = list(poc.PLAID_LAYOUTS) if args.layout == "all" else [args.layout]
    silhouettes = {"on": [True], "off": [False], "all": [False, True]}[args.silhouette]
    print(
        f"{FRAME_SIZE[0]}x{FRAME_SIZE[1]}, {args.frames} quadros, "
        f"OpenCV {cv2.__version__}, {cv2.getNumThreads()} threads"
    )
    header = f"{'perfil':7s} {'malha':8s} {'silh.':5s} {'linhas':>6s} {'pulsos':>6s} {'proc. ms':>9s} {'p95':>6s}"
    if args.display:
        header += f" {'tela ms':>8s}"
    print(header + f" {'FPS max':>8s}")
    for profile, layout, silhouette in itertools.product(profiles, layouts, silhouettes):
        result = run(frames, profile, layout, silhouette, args.display)
        row = (
            f"{profile:7s} {layout:8s} {'sim' if silhouette else 'nao':5s} "
            f"{result['lines']:6d} {result['pulses']:6d} "
            f"{result['process']:9.1f} {result['process_p95']:6.1f}"
        )
        if args.display:
            row += f" {result['display']:8.1f}"
        print(row + f" {result['fps']:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

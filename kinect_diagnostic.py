#!/usr/bin/env python3
"""Diagnostico de estabilidade do Kinect v1: python kinect_diagnostic.py [segundos]."""

import glob
import os
import re
import subprocess
import sys
import time

KINECT_PRODUCTS = {
    "02ae": "Camera",
    "02ad": "Audio",
    "02c2": "Hub (Kinect for Windows)",
    "02b0": "Motor",
}
LOG_PATTERNS = {
    "pacotes perdidos": r"Lost \d+ (?:total )?packets",
    "cabecalho invalido": r"Invalid magic",
    "falha de controle USB": r"transfer failed",
    "falha ao abrir/inicializar": r"Can't open device|Failed to fetch registration|Invalid index",
}
MIN_FPS = 25.0
MAX_LOG_EVENTS = 5
START_LIMIT = 30.0
SLOW_START = 3.0
SETTLE_SECONDS = 2.0


def list_usb_devices() -> bool:
    found = False
    for path in sorted(glob.glob("/sys/bus/usb/devices/*")):
        try:
            with open(f"{path}/idVendor") as file:
                vendor = file.read().strip()
            with open(f"{path}/idProduct") as file:
                product = file.read().strip()
            with open(f"{path}/speed") as file:
                speed = file.read().strip()
        except OSError:
            continue
        if vendor != "045e" or product not in KINECT_PRODUCTS:
            continue
        found = True
        pci = re.findall(r"0000:[0-9a-f]{2}:[0-9a-f]{2}\.\d", os.path.realpath(path))
        driver = (
            os.path.basename(os.path.realpath(f"/sys/bus/pci/devices/{pci[0]}/driver"))
            if pci
            else "?"
        )
        name = KINECT_PRODUCTS[product]
        print(f"  {os.path.basename(path)}: {name} ({vendor}:{product}) {speed} Mb/s via {driver}")
    if not found:
        print("  nenhum dispositivo Kinect encontrado no USB")
    return found


def worker(seconds: float) -> None:
    import freenect

    start = time.perf_counter()
    first = []

    def on_frame(kind: str):
        def callback(dev, data, timestamp):
            now = time.perf_counter() - start
            if not first:
                first.append(now)
            print(kind, f"{now:.3f}", flush=True)

        return callback

    def body(dev, ctx):
        now = time.perf_counter() - start
        if (not first and now > START_LIMIT) or (first and now - first[0] > seconds):
            raise freenect.Kill

    freenect.runloop(depth=on_frame("D"), video=on_frame("V"), body=body)


def run_capture(seconds: float) -> tuple[dict, str, bool]:
    # Subprocesso: chamadas nativas bloqueadas so podem ser interrompidas matando o processo.
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--worker", str(seconds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    hung = False
    try:
        out, err = proc.communicate(timeout=seconds + START_LIMIT + 15)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        hung = True
    stamps = {"D": [], "V": []}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in stamps:
            stamps[parts[0]].append(float(parts[1]))
    # libfreenect escreve diagnosticos tanto em stdout quanto em stderr.
    return stamps, out + err, hung


def summarize(stamps: list[float]) -> tuple[float, float, float]:
    """Retorna (tempo ate o 1o frame, fps em regime, maior intervalo em ms)."""
    steady = [t for t in stamps if t >= stamps[0] + SETTLE_SECONDS]
    if len(steady) < 2:
        return stamps[0], 0.0, 0.0
    fps = (len(steady) - 1) / (steady[-1] - steady[0])
    longest_gap = max(b - a for a, b in zip(steady, steady[1:])) * 1000
    return stamps[0], fps, longest_gap


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(float(sys.argv[2]))
        return 0

    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    print("Dispositivos USB:")
    if not list_usb_devices():
        print("\nVEREDITO: FALHA - o sistema nao enxerga o Kinect (cabo, alimentacao 12V ou porta).")
        return 2

    print(f"\nCapturando por {seconds:.0f} s apos o primeiro frame (ate {START_LIMIT:.0f} s de espera)...")
    stamps, log, hung = run_capture(seconds)
    counts = {label: len(re.findall(regex, log)) for label, regex in LOG_PATTERNS.items()}

    if hung:
        print("  captura travou e o processo foi encerrado")
    if not stamps["D"] or not stamps["V"]:
        for label, count in counts.items():
            print(f"  {label}: {count}")
        print("\nVEREDITO: FALHA - nenhum stream utilizavel; reconecte o Kinect e troque de porta/cabo.")
        return 2

    results = {"depth": summarize(stamps["D"]), "rgb": summarize(stamps["V"])}
    for name, (first, fps, gap) in results.items():
        print(f"  {name}: 1o frame em {first:.1f} s | {fps:.1f} fps em regime | maior intervalo {gap:.0f} ms")
    for label, count in counts.items():
        print(f"  {label}: {count}")

    slowest_start = max(first for first, _, _ in results.values())
    lowest_fps = min(fps for _, fps, _ in results.values())
    if lowest_fps < MIN_FPS or counts["pacotes perdidos"] + counts["cabecalho invalido"] > MAX_LOG_EVENTS:
        print("\nVEREDITO: INSTAVEL - stream perde dados; suspeite de porta/controladora, cabo ou alimentacao.")
        return 1
    if slowest_start > SLOW_START:
        print(f"\nVEREDITO: OK - captura estavel, mas a partida demora {slowest_start:.1f} s ate o primeiro frame.")
    else:
        print("\nVEREDITO: OK - captura estavel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Gera as imagens de docs/xadrez.md e o PNG transparente do padrao xadrez.

    python docs/gerar_imagens_xadrez.py
    python docs/gerar_imagens_xadrez.py --video videos/MVI_1732.MOV   # + quadro com movimento

As imagens usam uma semente fixa, entao saem iguais a cada execucao. O quadro
com movimento so e gerado se houver um video de ensaio (a pasta videos/ nao
esta no repositorio).
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import dance_interactive_poc as poc  # noqa: E402

OUT_DIR = os.path.join(ROOT, "docs", "img")
SEED = 7


def make_processor(layout: str, hue: float = poc.DEFAULT_PLAID_HUE, silhouette: bool = False):
    processor = poc.VisionProcessor(
        "xadrez", plaid_layout=layout, plaid_hue=hue, silhouette=silhouette
    )
    # Mesmo sorteio (sett, fases e angulos) em todas as imagens.
    processor._rng = np.random.default_rng(SEED)
    processor._plaid_patterns = [
        (poc.make_madras_sett(processor._rng), tuple(processor._rng.uniform(0, 1, 2)), angle)
        for angle in poc.make_plaid_angle_offsets(processor._rng, poc.PLAID_PATTERN_COUNT)
    ]
    return processor


def render_rest(
    layout: str, size: tuple[int, int], hue: float = poc.DEFAULT_PLAID_HUE, coverage: bool = False
) -> np.ndarray:
    """O padrao parado, sem pulsos, em fundo preto.

    Com coverage=True, todas as linhas saem brancas e com brilho pleno: o resultado
    e quanto cada pixel e coberto por linha (a mascara de opacidade do PNG).
    """
    width, height = size
    processor = make_processor(layout, hue)
    processor.pulse_rate = 0.0
    black = np.zeros((height, width, 3), np.uint8)
    processor.process(black)
    if not coverage:
        return processor.process(black).body_frame
    for field, _ in processor._field._layers:
        field.brightness = 1.0
    palette = poc.plaid_palette
    poc.plaid_palette = lambda hue: [(255, 255, 255)] * len(poc.PLAID_TINT_LEVELS)
    try:
        return processor.process(black).body_frame
    finally:
        poc.plaid_palette = palette


def to_transparent(image: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """BGR sobre preto + cobertura -> BGRA com as linhas opacas na cor real.

    Sobre preto, cada pixel vale cor * cobertura; dividir pela cobertura devolve
    a cor da linha, e a cobertura vira a opacidade: as bordas anti-aliased se
    misturam certo sobre qualquer fundo. (Usar o brilho como opacidade deixaria
    as linhas escuras transparentes e as claras sumiriam sobre fundo branco.)
    """
    alpha = coverage.max(axis=2)
    color = np.zeros_like(image)
    visible = alpha > 0
    color[visible] = np.clip(
        image[visible].astype(np.float32) * 255.0 / alpha[visible, None], 0, 255
    ).astype(np.uint8)
    return np.dstack((color, alpha))


def render_motion(video: str, size: tuple[int, int]) -> np.ndarray:
    """Quadro do xadrez depois de alguns segundos de ensaio: linhas, pulsos, silhueta."""
    width, height = size
    processor = make_processor("mix", silhouette=True)
    ticks = itertools.count()
    processor.clock = lambda: next(ticks) / 30.0
    capture = cv2.VideoCapture(video)
    result = None
    for index in range(600):
        ok, frame = capture.read()
        if not ok:
            break
        if index % 2:
            continue  # o video e 60 fps; um quadro em dois simula a captura a 30 fps
        result = processor.process(cv2.resize(frame, (width, height)))
    if result is None:
        raise RuntimeError(f"Nao foi possivel ler {video}")
    return result.body_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", help="video de ensaio para o quadro com movimento")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    size = (args.width, args.height)
    os.makedirs(OUT_DIR, exist_ok=True)

    mix = render_rest("mix", size)
    outputs = {
        "xadrez_transparente.png": to_transparent(mix, render_rest("mix", size, coverage=True)),
        "xadrez_mix.png": mix,
        "xadrez_regular.png": render_rest("regular", size),
    }
    hues = [render_rest("mix", (640, 360), hue) for hue in (30, 190, 300)]
    outputs["xadrez_matizes.png"] = np.hstack(hues)
    if args.video:
        outputs["xadrez_ensaio.png"] = render_motion(args.video, (1280, 720))
    for name, image in outputs.items():
        path = os.path.join(OUT_DIR, name)
        cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"{path}  {image.shape[1]}x{image.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

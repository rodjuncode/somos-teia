#!/usr/bin/env python3
"""Gera as imagens de docs/paisagens.md.

    python docs/gerar_imagens_paisagens.py                              # so o acervo
    python docs/gerar_imagens_paisagens.py --video videos/MVI_1732.MOV   # + borrao, feixes,
                                                                        #   paletas, regeneracao
                                                                        #   e silhueta

As imagens usam uma semente fixa. As que mostram o borrao precisam de um video de
ensaio (a pasta videos/ nao esta no repositorio); so a saida da PoC e salva, nunca
os quadros da camera.
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
SEED = 3
SIZE = (1280, 720)
PAINTING = os.path.join(poc.LANDSCAPE_DEFAULT_IMAGE, "aurelio-de-figueiredo-paisagem-1898.jpg")
CAPTURE_FPS = 30.0


def make_processor(**options) -> poc.VisionProcessor:
    processor = poc.VisionProcessor("paisagens", **options)
    processor._rng = np.random.default_rng(SEED)
    ticks = itertools.count()
    processor.clock = lambda: next(ticks) / CAPTURE_FPS
    return processor


def rehearsal_frames(video: str, seconds: float) -> list[np.ndarray]:
    """Um quadro em dois do video de 60 fps: simula a captura a 30 fps."""
    capture = cv2.VideoCapture(video)
    frames = []
    while len(frames) < seconds * CAPTURE_FPS:
        ok, frame = capture.read()
        if not ok:
            break
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, SIZE))
    if not frames:
        raise RuntimeError(f"Nao foi possivel ler {video}")
    return frames


def smudged(frames: list[np.ndarray], **options) -> tuple[poc.VisionProcessor, np.ndarray]:
    processor = make_processor(landscape_image=PAINTING, **options)
    result = None
    for frame in frames:
        result = processor.process(frame)
    return processor, result.body_frame


def label(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (12 + 11 * len(text), 30), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return image


def grid(images: list[np.ndarray], columns: int, size: tuple[int, int]) -> np.ndarray:
    tiles = [cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in images]
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[index : index + columns]) for index in range(0, len(tiles), columns)]
    return np.vstack(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", help="video de ensaio para as imagens com borrao")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    outputs = {}

    # Acervo: cada pintura como ela cobre a tela (recorte central, sem sorteio).
    names = poc.list_landscape_images(poc.LANDSCAPE_DEFAULT_IMAGE)
    framed = []
    for path in names:
        image = poc.fit_cover(cv2.imread(path), *SIZE, 0.5)
        framed.append(label(image, os.path.splitext(os.path.basename(path))[0]))
    outputs["paisagens_acervo.jpg"] = grid(framed, 4, (480, 270))

    if args.video:
        frames = rehearsal_frames(args.video, 20.0)
        processor, after = smudged(frames)
        before = processor._field._original
        outputs["paisagens_antes_depois.jpg"] = np.hstack(
            (label(before, "original"), label(after, "depois de 20 s de ensaio"))
        )

        crop = (slice(180, 540), slice(320, 960))
        widths = []
        for width in (1, 2, 6, 12):
            _, image = smudged(frames, landscape_bristle_width=width)
            widths.append(label(image[crop], f"feixe {width} px"))
        outputs["paisagens_feixes.jpg"] = grid(widths, 2, (640, 360))

        palettes = []
        for name in poc.LANDSCAPE_PALETTES:
            processor.landscape_palette = name
            palettes.append(label(processor.process(frames[-1]).body_frame, name))
        processor.landscape_palette = "original"
        outputs["paisagens_paletas.jpg"] = grid(palettes, 3, (640, 360))

        # Regeneracao: depois do ensaio, a cena fica parada (o ultimo quadro repetido).
        moments = {0: label(after, "fim do ensaio")}
        still = frames[-1]
        for frame_index in range(1, int(16 * CAPTURE_FPS) + 1):
            result = processor.process(still)
            seconds = frame_index / CAPTURE_FPS
            if seconds in (4.0, 8.0, 16.0):
                moments[seconds] = label(result.body_frame, f"{seconds:.0f} s parado")
        outputs["paisagens_regeneracao.jpg"] = grid(list(moments.values()), 4, (480, 270))

        # Com a pasta inteira, a janela mostra a pintura seguinte da lista. A
        # mascara e a figura sintetica do pi_benchmark, como a do Kinect: no video
        # o MOG2 quase nao separa os dancarinos e a janela mal apareceria.
        import pi_benchmark

        folder = make_processor(silhouette=True)
        folder.landscape_index = names.index(PAINTING)
        figure = pi_benchmark.synthetic_frames(1)[0][1]
        mask = cv2.resize(figure, SIZE, interpolation=cv2.INTER_NEAREST)
        result = None
        for frame in frames[: int(10 * CAPTURE_FPS)]:
            result = folder.process(frame, mask)
        outputs["paisagens_silhueta.jpg"] = label(
            result.body_frame, "silhueta (mascara simulada): janela para a proxima pintura"
        )

    for name, image in outputs.items():
        path = os.path.join(OUT_DIR, name)
        # Fotos de pinturas: JPEG fica ~5x menor que PNG sem perda visivel.
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{path}  {image.shape[1]}x{image.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

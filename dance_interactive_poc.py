#!/usr/bin/env python3
"""PoC de danca interativa com Kinect v1, webcam USB ou arquivo de video.

Instalação no Ubuntu (para Kinect, instale primeiro as libs nativas):
    sudo apt-get install libfreenect-dev freenect python3-dev python3-venv build-essential
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    # Opcional: binding Kinect v1
    python -m pip install -r requirements-kinect.txt

Execute com `python dance_interactive_poc.py`. Pressione q para sair.
Metronomo 4/4 opcional em todos os modos (--metronome, --bpm): t liga/desliga,
- diminui e = (ou +) aumenta o andamento. Requer libportaudio2 e sounddevice.
Fonte: --source kinect|webcam|arquivo.mp4; pressione m para alternar as fontes configuradas.
Use --fallback-video para reserva quando Kinect e webcam falharem.
Algoritmo: --mode kinect|mog2|optical_flow|conexoes|capim|mato|paisagens|xadrez.
O padrao e o xadrez, laranja e com a silhueta dos corpos ligada.
Faixa inicial: --min-depth/--max-depth (mm). Atalhos Kinect: a/z diminuem/aumentam
Min Depth; s/x diminuem/aumentam Max Depth, em passos de 100 mm. Pressione b
com a sala vazia para capturar o fundo e recortar apenas o que se move a frente.
O modo Kinect usa profundidade em milimetros (DEPTH_MM).
Atalhos do modo xadrez: a/z diminuem/aumentam a densidade, s/x o ganho, d/c a
rotacao do padrao, f/v giram o matiz no disco de cores (anti-horario/horario) e g
liga/desliga a silhueta dos corpos na cor base (com Kinect, b calibra o fundo);
l alterna a malha entre regular e mix (malhas em alturas diferentes).
"""

from __future__ import annotations

import argparse
import colorsys
import functools
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Protocol

# Checado antes de tudo: em Python antigo o erro apareceria so mais abaixo, como
# um TypeError obscuro em @dataclass(slots=True).
if sys.version_info < (3, 10):
    sys.exit(
        f"Esta PoC precisa de Python 3.10 ou mais novo (encontrado {sys.version.split()[0]}). "
        "No Raspberry Pi, use o Raspberry Pi OS Bookworm (Python 3.11)."
    )
try:
    import cv2
    import numpy as np
except ImportError as exc:
    sys.exit(
        f"Pacote ausente: {exc.name or exc}. Ative o venv (source .venv/bin/activate) e instale "
        "as dependencias: python -m pip install -r requirements.txt (notebook/desktop) ou "
        "-r requirements-pi.txt (Raspberry Pi). Veja 'Primeiros passos' no README."
    )

# Medido: o 1o frame do Kinect v1 chega em ~7 s neste equipamento.
KINECT_START_TIMEOUT = 20.0
# 0 significa "sem leitura" no mapa de profundidade do Kinect v1.
DEFAULT_MIN_DEPTH_MM = 800
DEFAULT_MAX_DEPTH_MM = 3000
DEPTH_STEP_MM = 100
BACKGROUND_SAMPLE_COUNT = 15
BACKGROUND_DIFF_MM = 80
# Arquivos do YOLO ficam ao lado deste script, nao no diretorio de onde ele e
# chamado: assim rodar de outra pasta nao baixa nem exporta tudo de novo.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PT_MODEL = os.path.join(SCRIPT_DIR, "yolov8n-pose.pt")
# Modelos OpenVINO retangulares: 320x240 recebe so 16 linhas de borda, nao 80.
YOLO_OPENVINO_INPUT = (256, 320)
YOLO_OPENVINO_MODEL = os.path.join(SCRIPT_DIR, "yolov8n-pose_256x320_openvino_model")
YOLO_OPENVINO_INT8_MODEL = os.path.join(SCRIPT_DIR, "yolov8n-pose_256x320_int8_openvino_model")
YOLO_INT8_CALIBRATION_DATA = "coco8-pose.yaml"
# Entrada do YOLO (altura, largura) e limite de pessoas por frame.
YOLO_INPUT_SIZE = (240, 320)
YOLO_MAX_DETECTIONS = 16
YOLO_MIN_KEYPOINT_CONFIDENCE = 0.25
# Mesmos limiares padrao do Ultralytics para deteccao de pessoas.
YOLO_DETECTION_CONFIDENCE = 0.25
YOLO_NMS_IOU = 0.7
DEFAULT_MAX_CONNECTIONS = 5
EDGE_STYLES = ("line", "bezier")
# Espessura fixa das ligacoes YOLO.
MESH_THICKNESS = 3
# Pontos de controle Bezier, relativos a aresta a->b: posicao ao longo dela
# (fracao do comprimento) e desvio perpendicular maximo (fracao do comprimento).
BEZIER_CONTROL_SPANS = ((0.15, 0.45), (0.55, 0.85))
BEZIER_MAX_BEND = 0.5
BEZIER_SAMPLES = 20


def bezier_basis(samples: int) -> np.ndarray:
    """Pesos (samples, 4) de uma Bezier cubica em t uniforme de 0 a 1."""
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
    return np.hstack(((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3))


BEZIER_BASIS = bezier_basis(BEZIER_SAMPLES)
# Modo capim: hastes de capim por 100 px de largura, altura maxima (fracao da altura
# do quadro) e ganho do empurrao do optical flow.
DEFAULT_GRASS_DENSITY = 25.0
DEFAULT_GRASS_HEIGHT = 0.5
DEFAULT_GRASS_GAIN = 1.0
# As bases ficam no sexto mais baixo do quadro; a menor haste tem metade da maior.
GRASS_BASE_BAND = 1.0 / 6.0
GRASS_MIN_HEIGHT_RATIO = 0.5
# Forma de repouso sorteada (fracoes da altura): inclinacao da ponta e desvio
# extra dos dois controles intermediarios, sutis por padrao.
GRASS_REST_LEAN = 0.12
GRASS_REST_BEND = 0.06
# Mola amortecida do deslocamento horizontal da ponta, por frame: o empurrao e
# GRASS_FLOW_PUSH * ganho * fluxo (px/frame); no maximo GRASS_MAX_BEND da altura.
GRASS_FLOW_PUSH = 0.3
GRASS_STIFFNESS = 0.06
GRASS_DAMPING = 0.88
GRASS_MAX_BEND = 0.8
# Alturas dos controles P1, P2 e ponta, e quanto do deslocamento da ponta cada um
# recebe (perfil ~t^2 de uma haste presa na base).
GRASS_CONTROL_LEVELS = np.array((1 / 3, 2 / 3, 1.0), dtype=np.float32)
GRASS_BEND_PROFILE = GRASS_CONTROL_LEVELS**2
# Peso do fluxo amostrado em P1, P2 e ponta no empurrao da haste.
GRASS_FORCE_WEIGHTS = np.array((0.2, 0.3, 0.5), dtype=np.float32)
GRASS_SAMPLES = 12
GRASS_BASIS = bezier_basis(GRASS_SAMPLES)
GRASS_THICKNESS = 1
# Tons de verde (BGR); cada cor e um unico cv2.polylines para todas as suas hastes.
GRASS_COLORS = (
    (70, 200, 90),
    (40, 170, 70),
    (110, 230, 140),
    (60, 210, 170),
    (50, 150, 110),
)
# Modo xadrez: padrao Madras de faixas coloridas que se cruzam em xadrez irregular.
# Comprimentos em fracao da altura do quadro, divididos por --density.
DEFAULT_PLAID_DENSITY = 1.0
DEFAULT_PLAID_GAIN = 1.0
DEFAULT_PLAID_ROTATION = 0.0
# Matiz inicial do xadrez: laranja (30 graus). None (so pela API) sorteia um.
DEFAULT_PLAID_HUE = 30.0
PLAID_HUE_STEP = 10.0
# Teclas do modo xadrez, em pares diminui/aumenta como as do Kinect: densidade
# muda por fator; ganho, rotacao e matiz (graus) por soma. Matiz crescente
# (vermelho -> amarelo -> verde -> azul) e o sentido horario no disco de cores.
PLAID_DENSITY_STEP = 1.1
PLAID_GAIN_STEP = 0.1
PLAID_ROTATION_STEP = 5.0
PLAID_KEY_ADJUSTMENTS = {
    ord("a"): {"density_factor": 1.0 / PLAID_DENSITY_STEP},
    ord("z"): {"density_factor": PLAID_DENSITY_STEP},
    ord("s"): {"gain_delta": -PLAID_GAIN_STEP},
    ord("x"): {"gain_delta": PLAID_GAIN_STEP},
    ord("d"): {"rotation_delta": -PLAID_ROTATION_STEP},
    ord("c"): {"rotation_delta": PLAID_ROTATION_STEP},
    ord("f"): {"hue_delta": -PLAID_HUE_STEP},
    ord("v"): {"hue_delta": PLAID_HUE_STEP},
    ord("h"): {"sensitivity_factor": 1.0 / 1.25},
    ord("n"): {"sensitivity_factor": 1.25},
}
# Acima de ~4x o numero de linhas (e o custo do desenho) passa de ~4x o padrao.
PLAID_DENSITY_RANGE = (0.25, 4.0)
# No mix, densidade maxima: acima disso a camada distante vira textura e o custo
# do desenho dispara (ela ja tem o dobro da densidade global).
PLAID_MIX_MAX_DENSITY = 2.0
PLAID_MAX_GAIN = 5.0
# Sensibilidade da interacao dos corpos com as linhas (--sensitivity, teclas h/n):
# quanto movimento basta, nao com que forca empurra (isso e o --gain). Divide o
# movimento minimo que o fluxo precisa ter para empurrar as linhas e o que a mao
# precisa andar para disparar pulsos, e multiplica o raio de toque da mao. O filtro
# do movimento fantasma do Farneback nao muda.
DEFAULT_PLAID_SENSITIVITY = 1.0
PLAID_SENSITIVITY_RANGE = (0.25, 4.0)
# Meio sett (a outra metade e o espelho, como nos tecidos Madras).
PLAID_HALF_SETT = 0.45
# Linhas por faixa e seus pesos; espacamento entre linhas da mesma faixa.
PLAID_BAND_LINES = (1, 2, 3, 4, 6)
PLAID_BAND_LINE_WEIGHTS = (0.3, 0.25, 0.2, 0.15, 0.1)
PLAID_LINE_SPACING = (0.006, 0.013)
# Vaos entre faixas: metade pequenos, metade grandes, para quadros desiguais.
PLAID_SMALL_GAP = (0.012, 0.035)
PLAID_LARGE_GAP = (0.06, 0.16)
PLAID_THICKNESSES = (1, 1, 1, 2, 3)
# Cores: um unico matiz base (--hue, em graus), variando so em luminosidade. Cada
# faixa sorteia um nivel de mistura com branco, da cor pura (0) ao branco (1).
PLAID_TINT_LEVELS = np.linspace(0.0, 1.0, 6)
# Disposicao das malhas do xadrez (tecla l alterna): "regular" e uma malha so;
# "mix" sao malhas independentes em alturas diferentes, da mais distante a mais
# proxima. Cada camada usa um padrao proprio (indice em PLAID_PATTERN_COUNT) e
# multiplica a densidade global, o brilho, o empurrao do fluxo (paralaxe: perto
# move mais) e soma a espessura das linhas. A camada do meio e a malha regular.
PlaidDepth = tuple[float, float, float, int]
PLAID_PATTERN_COUNT = 3
PLAID_REGULAR_PATTERN = 1
# Cada padrao alem do regular tem um desvio de angulo sorteado, somado a rotacao
# global: com todas as camadas no mesmo angulo as linhas ficam paralelas e o mix
# so parece mais denso. Como o xadrez se repete a cada 90 graus, os desvios ficam
# a pelo menos PLAID_MIX_MIN_ANGLE graus entre si (modulo 90), e do regular.
PLAID_MIX_MIN_ANGLE = 20.0
PLAID_LAYOUTS: dict[str, tuple[tuple[int, PlaidDepth], ...]] = {
    "regular": ((PLAID_REGULAR_PATTERN, (1.0, 1.0, 1.0, 0)),),
    "mix": (
        (0, (2.0, 0.4, 0.5, -1)),
        (PLAID_REGULAR_PATTERN, (1.0, 0.7, 1.0, 0)),
        (2, (0.55, 1.0, 1.6, 1)),
    ),
}
LAYOUT_TOGGLE_KEY = ord("l")
# Silhueta dos corpos (tecla g liga/desliga): no xadrez e preenchida pela cor base;
# no paisagens e uma janela para a proxima pintura.
# Fonte: mascara de profundidade do Kinect quando ha uma; senao, MOG2 em
# MOG2_SCALE. A mascara e limpa e suavizada na escala reduzida e ampliada
# com interpolacao linear: as bordas saem macias sem custo em resolucao cheia.
SILHOUETTE_TOGGLE_KEY = ord("g")
SILHOUETTE_SCALE = 0.25
SILHOUETTE_BLUR = 5
# Pulsos de neon que percorrem as linhas do xadrez como informacao: nascem em
# intervalos aleatorios (exponenciais, --pulses por segundo em media), numa linha
# e ponta sorteadas, e levam de PULSE_CROSSING_SECONDS para atravessa-la; no
# maximo PlaidRender.pulse_max ao mesmo tempo.
DEFAULT_PULSE_RATE = 4.0
PULSE_CROSSING_SECONDS = (1.0, 6.0)
# Cauda: o rastro dos ultimos PULSE_TAIL_SECONDS (entre os limites, em amostras
# da curva), desenhado em PULSE_FADE_STEPS trechos que apagam ate a ponta de tras.
PULSE_TAIL_SECONDS = 0.12
PULSE_TAIL_SAMPLES = (3, 40)
PULSE_FADE_STEPS = 4
# Pontos da cauda por nivel de brilho (a curva entre eles vira segmento reto).
PULSE_STEP_POINTS = 3
# Nucleo claro (mistura do matiz com branco) e halo do matiz puro, desenhado em
# 1/PULSE_GLOW_DOWNSCALE da resolucao, borrado e somado ao quadro.
PULSE_CORE_TINT = 0.7
PULSE_CORE_THICKNESS = 2
PULSE_GLOW_DOWNSCALE = 4
PULSE_GLOW_THICKNESS = 2
PULSE_GLOW_BLUR = 7
PULSE_GLOW_GAIN = 2.0
# Pulsos das maos (padrao da CLI): em vez de sorteados, os pulsos nascem onde uma
# mao passa por uma linha. O YOLO-pose acha os pulsos (punhos) de cada pessoa numa
# thread propria; a mao fica HAND_REACH do antebraco alem do punho (direcao
# cotovelo -> punho). Quando a mao, em movimento, passa a ate HAND_TOUCH_RADIUS
# (fracao da altura do quadro) de uma linha, nasce ali um pulso para cada lado.
# Para so gestos dispararem e a tela nao inundar (sete pessoas mexendo os bracos
# davam ~240 pulsos/s sem isso): cada mao dispara uma linha por vez e espera
# HAND_COOLDOWN s; cada linha espera HAND_LINE_COOLDOWN s; e no total nascem no
# maximo HAND_LINES_PER_SECOND linhas disparadas por segundo.
PULSE_SOURCES = ("hands", "random")
HAND_REACH = 0.35
HAND_TOUCH_RADIUS = 0.025
# Entre duas deteccoes, a mesma mao anda ate HAND_MAX_JUMP e precisa andar ao menos
# HAND_MIN_MOVE (fracoes da altura) para contar como gesto: o punho do YOLO treme
# ~0,5-1% da altura mesmo parado.
HAND_MAX_JUMP = 0.2
HAND_MIN_MOVE = 0.02
HAND_COOLDOWN = 0.25
HAND_LINE_COOLDOWN = 1.0
HAND_LINES_PER_SECOND = 12.0
COCO_ELBOWS_WRISTS = ((7, 9), (8, 10))
# Coordenadas subpixel no cv2.polylines (2 bits = 1/4 px): o pulso desliza suave.
PULSE_SUBPIXEL_SHIFT = 2
# Cada linha e uma cadeia de nos a cada PLAID_NODE_SPACING px, desenhada como
# Catmull-Rom (Beziers cubicas encadeadas) com PLAID_SEGMENT_SAMPLES por trecho.
# As pontas ficam PLAID_NODE_SPACING fora do quadro e nao se movem.
PLAID_NODE_SPACING = 40.0
PLAID_SEGMENT_SAMPLES = 6
PLAID_SEGMENT_BASIS = bezier_basis(PLAID_SEGMENT_SAMPLES + 1)[:-1]


@dataclass(frozen=True)
class PlaidRender:
    """Qualidade do desenho do xadrez; --pi troca o padrao por PI_PLAID_RENDER."""

    # Sem anti-aliasing as linhas grossas saem direto do cv2.polylines (sem a
    # camada + dilate + max) e cada linha custa ~1/3.
    antialias: bool = True
    segment_samples: int = PLAID_SEGMENT_SAMPLES
    pulse_max: int = 150
    soft_silhouette: bool = True
    # Compoe o quadro de debug (camera + desenho); sem janela de debug, nao precisa.
    debug_output: bool = True

    @property
    def line_type(self) -> int:
        return cv2.LINE_AA if self.antialias else cv2.LINE_8


# Raspberry Pi 4: sem anti-aliasing nem bordas suaves na silhueta, curvas com
# metade das amostras (as linhas so curvam onde o fluxo as empurra), menos pulsos
# simultaneos e sem quadro de debug.
# Modos aceitos pelo --pi; o primeiro e o padrao.
PI_MODES = ("xadrez", "paisagens")
# Intervalo, em segundos, entre os relatorios de FPS no terminal com --pi.
PI_STATUS_INTERVAL = 5.0
PI_PLAID_RENDER = PlaidRender(
    antialias=False,
    segment_samples=3,
    pulse_max=40,
    soft_silhouette=False,
    debug_output=False,
)
# Mola por no: empurrao do fluxo, retorno ao repouso, tensao entre vizinhos (a
# linha se comporta como uma corda esticada), amortecimento e deslocamento maximo.
PLAID_FLOW_PUSH = 0.3
PLAID_STIFFNESS = 0.04
PLAID_TENSION = 0.15
PLAID_DAMPING = 0.85
PLAID_MAX_DISPLACEMENT = 90.0
# Modo mato: tracos retos e finos espalhados pela tela, numa estetica quase
# impressionista. Tres camadas, da mais distante a mais proxima: a base, estatica,
# desenhada uma vez; a do meio, que vive com os corpos e e empurrada pelo fluxo;
# e uma frente estatica, por cima de tudo. Por camada: quantidade para um quadro
# 1280x720 (escalada pela area e por --density), brilho e faixa de comprimento dos
# tracos (fracao da altura do quadro): quanto mais perto, mais longos.
MEADOW_LAYERS = {
    "base": (35000, 0.7, (0.006, 0.03)),
    "middle": (2000, 0.9, (0.03, 0.10)),
    "front": (600, 1.0, (0.06, 0.18)),
}
MEADOW_THICKNESS = 2
# Paleta (cor, peso): predominam verdes, do verde bandeira ao verde agua, com
# alguns marrons e poucos rosas e roxos. Troque aqui para mudar a paisagem.
MEADOW_PALETTE = (
    ("#00552E", 8), ("#009C3B", 10), ("#1E8449", 9), ("#2E9E4F", 9), ("#4CAF50", 8),
    ("#7CB342", 6), ("#3CB371", 9), ("#20B2AA", 7), ("#48C9B0", 7), ("#66CDAA", 7),
    ("#5D4037", 4), ("#795548", 4), ("#8D6E63", 4),
    ("#E91E63", 1), ("#F48FB1", 1),
    ("#8E44AD", 1), ("#6A1B9A", 1),
)
# Tracos do meio: o fluxo nas duas pontas arrasta o centro e a diferenca entre
# elas o gira (o traco nao estica). Nao ha mola: o que os dancarinos mudam fica
# mudado. O atrito para o traco quando o empurrao acaba; com o empurrao igual a
# 1 - atrito, um traco tocado anda na velocidade do corpo (com --gain 1) e gira
# no ritmo que o fluxo sugere (giro = d x (f1 - f0) / 2|d|^2).
MEADOW_DAMPING = 0.85
MEADOW_FLOW_PUSH = 1.0 - MEADOW_DAMPING
MEADOW_TURN_PUSH = (1.0 - MEADOW_DAMPING) / 2.0
# Modo paisagens: uma imagem cobre a tela e o fluxo dos corpos a borra e espalha.
# A imagem e ampliada ate cobrir o quadro sem distorcer; o excesso e cortado e
# --image-anchor escolhe a faixa visivel (0 = topo, 0,5 = meio, 1 = base).
# --image aceita uma pasta (todas as imagens, em ordem alfabetica; teclas a/z
# alternam entre elas) ou um arquivo.
LANDSCAPE_DEFAULT_IMAGE = os.path.join(SCRIPT_DIR, "paisagens")
LANDSCAPE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
# a volta para a imagem anterior e z avanca para a proxima (o mesmo par das outras teclas).
LANDSCAPE_IMAGE_KEYS = {ord("a"): -1, ord("z"): 1}
# Vale para imagens em paisagem; nas em retrato a altura do recorte e sorteada
# (0 a 1) toda vez que a imagem e carregada.
LANDSCAPE_DEFAULT_ANCHOR = 0.5
# Suavizacao do campo de fluxo reduzido antes de ampliar (impar, em celulas):
# sem ela o borrao herda os blocos de 8x8 px do Farneback.
LANDSCAPE_FLOW_BLUR = 5
# Com --gain 1 a tinta anda a esta fracao da velocidade do corpo: a pleno, sete
# pessoas deixavam a imagem irreconhecivel em ~5 s.
LANDSCAPE_FLOW_SCALE = 0.4
# "Cerdas": ruido fixo e fino que varia quanto cada pixel e arrastado (1 +/-
# LANDSCAPE_BRISTLE desvios). Linhas vizinhas andam diferente e o rastro vira
# estrias ao longo do gesto, como pinceladas, em vez de um marmorizado liso.
LANDSCAPE_BRISTLE = 0.8
# Espessura dos feixes (px do quadro processado): o ruido e borrado com sigma =
# metade dela. 1-2 px da estrias finas; ~6 px, pinceladas largas; ~12 px, manchas
# quase lisas. Teclas h/n afinam/engrossam; mudar refaz as cerdas (~15-25 ms).
LANDSCAPE_BRISTLE_WIDTH = 2.0
LANDSCAPE_BRISTLE_WIDTH_RANGE = (1.0, 16.0)
LANDSCAPE_BRISTLE_KEYS = {ord("h"): -1.0, ord("n"): 1.0}
# Paletas neon (estetica cyberpunk) por mapa de gradiente: o brilho de cada pixel
# vira uma cor do gradiente, do escuro ao claro. E aplicado no desenho, nao na
# tela guardada: trocar de paleta e instantaneo e mantem borrao, regeneracao e a
# janela da silhueta. O brilho de cada pintura e esticado entre os percentis
# LANDSCAPE_PALETTE_STRETCH para ocupar o gradiente todo (as pinturas sao escuras).
# Tecla p alterna; "original" mostra as cores da pintura.
LANDSCAPE_PALETTES: dict[str, Optional[tuple[str, ...]]] = {
    "original": None,
    "synthwave": ("#0d0221", "#2a0a4a", "#7b1fa2", "#ff2a6d", "#ff9e3d", "#fff5c0"),
    "cyberpunk": ("#05020f", "#1b0b3b", "#5a189a", "#f107a3", "#00f0ff", "#fcee09"),
    "matrix": ("#000000", "#001a06", "#00500f", "#00a82d", "#39ff6a", "#e0ffe8"),
    "tron": ("#000306", "#04121f", "#0a3a5c", "#008cff", "#00e5ff", "#eaffff"),
    "vaporwave": ("#140a2b", "#3d1d6b", "#ff71ce", "#b967ff", "#01cdfe", "#fffb96"),
}
LANDSCAPE_PALETTE_STRETCH = (2.0, 98.0)
LANDSCAPE_PALETTE_KEY = ord("p")
# Como calcular os mapas do remap: "opencv" (cv2.multiply/scaleAdd, ~20% mais
# rapido em 720p) ou "numpy" (a versao original). O arredondamento do OpenCV
# difere em ate ~1e-4 px; num ensaio de 400 quadros, 13 de 921 600 pixels
# terminaram 1 nivel (de 255) diferentes. Troque aqui para voltar a versao NumPy.
LANDSCAPE_SMEAR_BACKEND = "opencv"
# Regeneracao: onde ninguem age ha LANDSCAPE_REGEN_DELAY s, a pintura original
# volta aos poucos (cross-fade) e fica ~completa em LANDSCAPE_REGEN_TIME s (resta
# LANDSCAPE_REGEN_RESIDUE da diferenca). O tempo parado e contado na grade do
# fluxo (1/8 do quadro); a volta entra em LANDSCAPE_REGEN_RAMP s, sem degrau.
# Teclas: d/c diminuem/aumentam a espera e f/v o tempo de volta.
LANDSCAPE_REGEN_DELAY = 3.0
LANDSCAPE_REGEN_TIME = 20.0
LANDSCAPE_REGEN_RESIDUE = 0.02
LANDSCAPE_REGEN_RAMP = 1.0
LANDSCAPE_REGEN_DELAY_RANGE = (0.0, 60.0)
LANDSCAPE_REGEN_TIME_RANGE = (2.0, 300.0)
LANDSCAPE_REGEN_KEYS = {
    ord("d"): {"delay_delta": -1.0},
    ord("c"): {"delay_delta": 1.0},
    ord("f"): {"time_delta": -5.0},
    ord("v"): {"time_delta": 5.0},
}
# A mistura em 8 bits arredonda passos pequenos para zero: a cada quadro (~3%) a
# pintura pararia a ~18 niveis do original. Aplicada a cada LANDSCAPE_REGEN_EVERY
# quadros, com o passo acumulado, ela para mais perto; o resto e copiado do
# original onde a diferenca ja e menor que o que o passo consegue mover.
LANDSCAPE_REGEN_EVERY = 8
# Teclas do modo paisagens, no mesmo par do ganho do xadrez: s diminui e x
# aumenta a forca do borrao (o --gain), de 0 (congela a imagem) a 5.
LANDSCAPE_GAIN_STEP = 0.1
LANDSCAPE_MAX_GAIN = 5.0
LANDSCAPE_KEY_ADJUSTMENTS = {ord("s"): -LANDSCAPE_GAIN_STEP, ord("x"): LANDSCAPE_GAIN_STEP}
# Metronomo (todos os modos): compasso 4/4, cabeca do compasso mais aguda e forte.
# Teclas: t liga/desliga; - diminui e = (ou +) aumenta o andamento.
METRONOME_DEFAULT_BPM = 60.0
METRONOME_BPM_RANGE = (30.0, 240.0)
METRONOME_BPM_STEP = 5.0
METRONOME_BEATS_PER_BAR = 4
METRONOME_SAMPLE_RATE = 48000
# Clique: seno com decaimento exponencial; (frequencia Hz, amplitude 0-1).
METRONOME_ACCENT = (1760.0, 0.8)
METRONOME_BEAT = (880.0, 0.45)
METRONOME_CLICK_SECONDS = 0.05
METRONOME_TOGGLE_KEY = ord("t")
METRONOME_TEMPO_KEYS = {ord("-"): -1.0, ord("="): 1.0, ord("+"): 1.0}
# Medido em H.264 720p60: 2 threads decodificam em ~4 ms/frame com ~40% da CPU
# gasta pelo padrao do FFmpeg (8 threads), deixando nucleos livres para a inferencia.
VIDEO_DECODE_THREADS = 2
# (frame BGR, profundidade, mascara); fontes 2D nao tem profundidade nem mascara.
FrameData = tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
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
                # cvtColor ja grava o RGB convertido em um array novo.
                depth = np.array(depth, copy=True)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                with self._lock:
                    min_depth, max_depth = self._min_depth, self._max_depth
                    background_depth = self._background_depth
                body_mask = make_kinect_body_mask(
                    depth, min_depth, max_depth, background_depth
                )
                with self._lock:
                    self._frame = (bgr, depth, body_mask)
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
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
                with self._lock:
                    self._frame = (bgr, None, None)
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
        self._capture = cv2.VideoCapture(
            self.path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, VIDEO_DECODE_THREADS]
        )
        if not self._capture.isOpened():
            self._capture.release()
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
                with self._lock:
                    self._frame = (bgr, None, None)
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
    body_mask: Optional[np.ndarray]
    debug_frame: np.ndarray
    body_frame: np.ndarray


def yolo_keypoints(backend, frame_bgr: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Keypoints COCO (pessoas, 17, 2) em coordenadas do quadro, e confiancas."""
    height, width = frame_bgr.shape[:2]
    input_height, input_width = YOLO_INPUT_SIZE
    frame_small = frame_bgr
    if width > 2 * input_width and height > 2 * input_height:
        # Reduzir ate 2x com LINEAR e terminar com AREA da quase o mesmo resultado
        # que AREA direto (diferenca media de 0,2 nivel) em ~40% do tempo, em 720p.
        frame_small = cv2.resize(
            frame_small,
            (2 * input_width, 2 * input_height),
            interpolation=cv2.INTER_LINEAR,
        )
    frame_small = cv2.resize(
        frame_small, (input_width, input_height), interpolation=cv2.INTER_AREA
    )
    points, confidence = backend.keypoints(frame_small)
    points = np.asarray(points, dtype=np.float32) * np.array(
        (width / input_width, height / input_height), dtype=np.float32
    )
    return points, confidence


def hand_positions(points: np.ndarray, confidence: Optional[np.ndarray]) -> np.ndarray:
    """Maos (N, 2): um pouco alem de cada punho visivel, na direcao do antebraco."""
    hands = []
    for person in range(len(points)):
        for elbow, wrist in COCO_ELBOWS_WRISTS:
            seen = confidence is None or confidence[person, wrist] >= YOLO_MIN_KEYPOINT_CONFIDENCE
            if not seen:
                continue
            hand = points[person, wrist].astype(np.float32)
            if confidence is None or confidence[person, elbow] >= YOLO_MIN_KEYPOINT_CONFIDENCE:
                hand = hand + HAND_REACH * (points[person, wrist] - points[person, elbow])
            hands.append(hand)
    return np.asarray(hands, dtype=np.float32).reshape(-1, 2)


def hand_paths(
    previous: np.ndarray, current: np.ndarray, max_jump: float, min_move: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Caminho (antes, agora) de cada mao que andou entre duas deteccoes.

    A ordem das pessoas do YOLO muda de um quadro para outro: cada mao atual e
    casada com a anterior mais proxima, a ate max_jump. Maos novas, que sumiram ou
    que andaram menos de min_move nao geram caminho.
    """
    if len(previous) == 0 or len(current) == 0:
        return []
    distance = np.hypot(*(current[:, None] - previous[None]).transpose(2, 0, 1))
    paths = []
    for index, nearest in enumerate(distance.argmin(axis=1)):
        if min_move <= distance[index, nearest] <= max_jump:
            paths.append((previous[nearest], current[index]))
    return paths


class HandTracker:
    """Roda o YOLO-pose numa thread propria e publica as maos mais recentes.

    O xadrez entrega cada quadro e segue sem esperar: o YOLO sempre pega o quadro
    mais novo, e o desenho usa a ultima deteccao pronta. Assim os ~15-20 ms de
    inferencia em CPU nao entram no tempo de cada quadro do xadrez.
    """

    def __init__(self, backend) -> None:
        self._backend = backend
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self.hands = np.empty((0, 2), dtype=np.float32)
        self.sequence = 0
        self.error: Optional[Exception] = None
        self._thread = threading.Thread(target=self._loop, name="hand-tracker", daemon=True)
        self._thread.start()

    def submit(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self._frame = frame_bgr
        self._new_frame.set()

    def latest(self) -> tuple[int, np.ndarray]:
        with self._lock:
            return self.sequence, self.hands

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._new_frame.wait(timeout=0.1):
                continue
            self._new_frame.clear()
            with self._lock:
                frame = self._frame
            try:
                hands = hand_positions(*yolo_keypoints(self._backend, frame))
            except Exception as exc:
                self.error = exc
                print(f"Aviso: deteccao das maos parou ({exc}).", file=sys.stderr)
                return
            with self._lock:
                self.hands = hands
                self.sequence += 1

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class PresentationDeadband:
    def __init__(self, threshold_px: float = 4.0) -> None:
        if threshold_px < 0:
            raise ValueError("point_deadband deve ser >= 0")
        self.threshold_px = threshold_px
        self._positions: dict[tuple[int, int], tuple[float, float]] = {}

    def reset(self) -> None:
        self._positions.clear()

    def apply(
        self, points: dict[tuple[int, int], tuple[float, float]]
    ) -> dict[tuple[int, int], tuple[float, float]]:
        threshold = self.threshold_px
        previous_positions = self._positions
        stabilized = {}
        for key, (x, y) in points.items():
            previous = previous_positions.get(key)
            if previous is not None and threshold > 0:
                dx, dy = x - previous[0], y - previous[1]
                distance = math.hypot(dx, dy)
                if distance <= threshold:
                    x, y = previous
                else:
                    step = (distance - threshold) / distance
                    x, y = previous[0] + dx * step, previous[1] + dy * step
            stabilized[key] = (x, y)
        self._positions = stabilized
        return stabilized


def export_yolo_openvino_model(
    yolo_class=None, int8: bool = False, calibration_data: Optional[str] = None
) -> str:
    target = YOLO_OPENVINO_INT8_MODEL if int8 else YOLO_OPENVINO_MODEL
    if os.path.isdir(target):
        return target
    if yolo_class is None:
        from ultralytics import YOLO as yolo_class
    model = yolo_class(YOLO_PT_MODEL)
    options = (
        {"quantize": 8, "data": calibration_data or YOLO_INT8_CALIBRATION_DATA}
        if int8
        else {}
    )
    exported = model.export(format="openvino", imgsz=YOLO_OPENVINO_INPUT, **options)
    if not exported or not os.path.isdir(exported):
        raise RuntimeError(f"A exportacao nao criou {target}")
    os.replace(exported, target)
    return target


class LetterboxPoseBackend:
    """Shared YOLO pose pre/post-processing; subclasses run the raw network."""

    def __init__(self, input_size: tuple[int, int]) -> None:
        self._input_size = input_size
        self._canvas = np.full((*input_size, 3), 114, dtype=np.uint8)

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        """Returns the (56, anchors) output: box, score and 17 x (x, y, conf)."""
        raise NotImplementedError

    def keypoints(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Returns (people, 17, 2) image coordinates and (people, 17) confidences."""
        height, width = frame_bgr.shape[:2]
        input_height, input_width = self._input_size
        # Letterbox centralizado, igual ao pre-processamento do Ultralytics.
        ratio = min(input_height / height, input_width / width)
        resized_width, resized_height = round(width * ratio), round(height * ratio)
        if (resized_width, resized_height) != (width, height):
            frame_bgr = cv2.resize(
                frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
            )
        top = round((input_height - resized_height) / 2 - 0.1)
        left = round((input_width - resized_width) / 2 - 0.1)
        self._canvas[:] = 114
        self._canvas[top : top + resized_height, left : left + resized_width] = frame_bgr
        output = self._infer(cv2.dnn.blobFromImage(self._canvas, 1.0 / 255.0, swapRB=True))

        empty = (
            np.empty((0, 17, 2), dtype=np.float32),
            np.empty((0, 17), dtype=np.float32),
        )
        candidates = output[:, output[4] > YOLO_DETECTION_CONFIDENCE].T
        if candidates.size == 0:
            return empty
        boxes = candidates[:, :4].copy()
        boxes[:, :2] -= boxes[:, 2:] / 2
        # top_k do OpenCV corta candidatos antes da supressao; o limite vem depois.
        kept = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            candidates[:, 4].tolist(),
            YOLO_DETECTION_CONFIDENCE,
            YOLO_NMS_IOU,
        )
        if len(kept) == 0:
            return empty
        kept = np.asarray(kept).reshape(-1)[:YOLO_MAX_DETECTIONS]
        keypoints = candidates[kept, 5:].reshape(-1, 17, 3)
        points = (keypoints[:, :, :2] - np.array((left, top), dtype=np.float32)) / ratio
        # O Ultralytics prende keypoints fora do quadro na borda; aqui tambem.
        np.clip(points[:, :, 0], 0, width, out=points[:, :, 0])
        np.clip(points[:, :, 1], 0, height, out=points[:, :, 1])
        return points, keypoints[:, :, 2]


class OpenVINOPoseBackend(LetterboxPoseBackend):
    """Runs an exported YOLO pose model on OpenVINO without Ultralytics overhead."""

    def __init__(self, model_dir: str) -> None:
        import openvino as ov

        xml_files = [name for name in os.listdir(model_dir) if name.endswith(".xml")]
        if len(xml_files) != 1:
            raise RuntimeError(f"Esperado um .xml em {model_dir}")
        compiled = ov.Core().compile_model(
            os.path.join(model_dir, xml_files[0]),
            "CPU",
            {"PERFORMANCE_HINT": "LATENCY"},
        )
        self._request = compiled.create_infer_request()
        _, _, input_height, input_width = (int(size) for size in compiled.inputs[0].shape)
        super().__init__((input_height, input_width))

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        self._request.infer({0: blob})
        return self._request.get_output_tensor(0).data[0]


class CudaGraphPoseBackend(LetterboxPoseBackend):
    """Replays the PyTorch pose network as one CUDA graph.

    Launching each layer from Python costs more CPU than the GPU spends on yolov8n;
    a captured graph replaces ~200 kernel launches with a single call.
    """

    def __init__(self, model, input_size: tuple[int, int] = YOLO_OPENVINO_INPUT) -> None:
        import torch

        super().__init__(input_size)
        self._torch = torch
        network = model.model.fuse().to("cuda").eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self._input = torch.zeros((1, 3, *input_size), device="cuda")
        self._host_input = torch.zeros((1, 3, *input_size)).pin_memory()
        with torch.inference_mode():
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    network(self._input)
            torch.cuda.current_stream().wait_stream(warmup_stream)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._output = network(self._input)[0][0]

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        self._host_input.numpy()[:] = blob
        self._input.copy_(self._host_input, non_blocking=True)
        self._graph.replay()
        return self._output.cpu().numpy()


class UltralyticsPoseBackend:
    """Runs an Ultralytics pose model and returns keypoints as NumPy arrays."""

    def __init__(self, model, device: Optional[int | str] = None) -> None:
        self._model = model
        self._options = {"device": device} if device is not None else {}

    @staticmethod
    def _to_numpy(value) -> Optional[np.ndarray]:
        if value is None:
            return None
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def keypoints(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Returns (people, 17, 2) image coordinates and (people, 17) confidences."""
        result = self._model(
            frame_bgr,
            imgsz=YOLO_INPUT_SIZE[1],
            max_det=YOLO_MAX_DETECTIONS,
            verbose=False,
            **self._options,
        )[0]
        keypoints = getattr(result, "keypoints", None)
        points = self._to_numpy(getattr(keypoints, "xy", None))
        if points is None:
            points = np.empty((0, 17, 2), dtype=np.float32)
        return points, self._to_numpy(getattr(keypoints, "conf", None))


def sample_flow(
    flow: np.ndarray, points: np.ndarray, scale: np.ndarray, threshold: float
) -> np.ndarray:
    """Fluxo em px/frame do quadro original nos pontos (..., 2) dados.

    flow e o campo reduzido em scale = (sx, sy); movimento abaixo de threshold
    (na escala reduzida) e ruido parado da camera, como na mascara do optical_flow.
    """
    flow_height, flow_width = flow.shape[:2]
    cells = (points * scale).astype(np.intp)
    np.clip(cells[..., 0], 0, flow_width - 1, out=cells[..., 0])
    np.clip(cells[..., 1], 0, flow_height - 1, out=cells[..., 1])
    sampled = flow[cells[..., 1], cells[..., 0]]
    sampled[np.hypot(sampled[..., 0], sampled[..., 1]) <= threshold] = 0.0
    return sampled / scale


def make_madras_sett(
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Sorteia um sett Madras espelhado.

    Retorna (periodo, posicoes, tons, espessuras); periodo e posicoes em fracao
    da altura do quadro, posicoes dentro de [0, periodo); tons sao indices de
    PLAID_TINT_LEVELS.
    """
    half = PLAID_HALF_SETT
    offsets, tints, thicknesses = [], [], []
    position = 0.0
    while position < half:
        tint = int(rng.integers(len(PLAID_TINT_LEVELS)))
        thickness = int(rng.choice(PLAID_THICKNESSES))
        count = int(rng.choice(PLAID_BAND_LINES, p=PLAID_BAND_LINE_WEIGHTS))
        spacing = rng.uniform(*PLAID_LINE_SPACING)
        for _ in range(count):
            if position >= half:
                break
            offsets.append(position)
            tints.append(tint)
            thicknesses.append(thickness)
            position += spacing
        gap = PLAID_SMALL_GAP if rng.random() < 0.5 else PLAID_LARGE_GAP
        position += rng.uniform(*gap)
    offsets = np.asarray(offsets)
    # O espelho comeca depois da ultima linha, para nao duplicar a do eixo.
    mirrored = 2.0 * half - offsets[::-1]
    keep = mirrored > offsets[-1]
    return (
        2.0 * half,
        np.concatenate((offsets, mirrored[keep])),
        np.concatenate((tints, np.asarray(tints)[::-1][keep])).astype(np.intp),
        np.concatenate((thicknesses, np.asarray(thicknesses)[::-1][keep])).astype(np.intp),
    )


def make_plaid_angle_offsets(rng: np.random.Generator, count: int) -> list[float]:
    """Desvios de angulo por padrao: 0 no regular, os outros bem separados (mod 90)."""
    offsets = [0.0]
    while len(offsets) < count:
        candidate = float(rng.uniform(0.0, 90.0))
        if all(
            min(abs(candidate - other) % 90.0, 90.0 - abs(candidate - other) % 90.0)
            >= PLAID_MIX_MIN_ANGLE
            for other in offsets
        ):
            offsets.append(candidate)
    # O regular fica na sua posicao, com desvio 0.
    offsets.insert(PLAID_REGULAR_PATTERN, offsets.pop(0))
    return offsets[:count]


def plaid_palette(hue: float) -> list[tuple[int, int, int]]:
    """Cores BGR de cada nivel de PLAID_TINT_LEVELS: o matiz puro misturado ao branco."""
    red, green, blue = colorsys.hsv_to_rgb((hue % 360.0) / 360.0, 1.0, 1.0)
    pure = np.array((blue, green, red))
    return [
        tuple(int(round(255.0 * channel)) for channel in pure + level * (1.0 - pure))
        for level in PLAID_TINT_LEVELS
    ]


class ElasticLines:
    """Linhas retas em repouso, cada uma uma cadeia de nos que o fluxo empurra.

    Todas as linhas tem o mesmo numero de nos. Cada no interno e uma mola 2D
    puxada de volta ao repouso e aos vizinhos; as pontas ficam fixas.
    """

    def __init__(
        self,
        rest: np.ndarray,
        tints: np.ndarray,
        thicknesses: np.ndarray,
        segment_samples: int = PLAID_SEGMENT_SAMPLES,
    ) -> None:
        self._segment_samples = segment_samples
        self._basis = (
            PLAID_SEGMENT_BASIS
            if segment_samples == PLAID_SEGMENT_SAMPLES
            else bezier_basis(segment_samples + 1)[:-1]
        )
        self._rest = rest.astype(np.float32)
        self._displacement = np.zeros_like(self._rest)
        self._velocity = np.zeros_like(self._rest)
        # (tom, espessura, indices das linhas): um cv2.polylines por grupo.
        self.groups = list(self._group_lines(tints, thicknesses))

    @staticmethod
    def _group_lines(tints: np.ndarray, thicknesses: np.ndarray):
        keys = tints * (int(thicknesses.max(initial=0)) + 1) + thicknesses
        for key in np.unique(keys):
            indices = np.flatnonzero(keys == key)
            yield int(tints[indices[0]]), int(thicknesses[indices[0]]), indices

    def __len__(self) -> int:
        return len(self._rest)

    @property
    def sample_count(self) -> int:
        """Pontos de cada curva devolvida por curves()."""
        return (self._rest.shape[1] - 1) * self._segment_samples + 1

    def reset(self) -> None:
        self._displacement[:] = 0.0
        self._velocity[:] = 0.0

    def update(
        self, flow: Optional[np.ndarray], scale: np.ndarray, threshold: float, gain: float
    ) -> None:
        if len(self) == 0:
            return
        displacement = self._displacement[:, 1:-1]
        velocity = self._velocity[:, 1:-1]
        velocity *= PLAID_DAMPING
        if flow is not None:
            points = self._rest[:, 1:-1] + displacement
            velocity += PLAID_FLOW_PUSH * gain * sample_flow(flow, points, scale, threshold)
        neighbors = self._displacement[:, :-2] + self._displacement[:, 2:]
        velocity += PLAID_TENSION * (neighbors - 2.0 * displacement)
        velocity -= PLAID_STIFFNESS * displacement
        displacement += velocity
        length = np.hypot(displacement[..., 0], displacement[..., 1])[..., None]
        np.multiply(
            displacement,
            np.minimum(1.0, PLAID_MAX_DISPLACEMENT / np.maximum(length, 1e-6)),
            out=displacement,
        )

    def curves(self) -> np.ndarray:
        """Pontos (linhas, amostras, 2) das curvas, em float32."""
        nodes = self._rest + self._displacement
        # Catmull-Rom como Bezier cubica: controles a 1/6 da tangente dos vizinhos.
        padded = np.concatenate(
            (2 * nodes[:, :1] - nodes[:, 1:2], nodes, 2 * nodes[:, -1:] - nodes[:, -2:-1]),
            axis=1,
        )
        start, end = nodes[:, :-1], nodes[:, 1:]
        anchors = np.stack(
            (
                start,
                start + (padded[:, 2:-1] - padded[:, :-3]) / 6.0,
                end - (padded[:, 3:] - padded[:, 1:-2]) / 6.0,
                end,
            ),
            axis=2,
        )
        curves = (self._basis @ anchors).reshape(len(self), -1, 2)
        return np.concatenate((curves, nodes[:, -1:]), axis=1)


class NeonPulses:
    """Pulsos de luz que atravessam as linhas do xadrez de ponta a ponta.

    A posicao de cada pulso e medida em amostras da curva da sua linha, entao ele
    acompanha a linha mesmo quando o fluxo a deforma.
    """

    def __init__(
        self,
        line_counts: list[int],
        sample_counts: list[int],
        rate: float,
        rng: np.random.Generator,
        size: tuple[int, int],
        render: PlaidRender = PlaidRender(),
    ) -> None:
        self.rate = rate
        self._rng = rng
        self._render = render
        self._line_counts = np.asarray(line_counts, dtype=np.intp)
        self._sample_counts = np.asarray(sample_counts, dtype=np.float32)
        width, height = size
        # Nucleo desenhado a parte e somado: por cima da linha, a cauda apagada
        # escureceria a linha em vez de so clarear.
        self._core = np.zeros((height, width, 3), dtype=np.uint8)
        self._glow = np.zeros(
            (max(1, height // PULSE_GLOW_DOWNSCALE), max(1, width // PULSE_GLOW_DOWNSCALE), 3),
            dtype=np.uint8,
        )
        self.clear()

    def __len__(self) -> int:
        return len(self._position)

    def clear(self) -> None:
        self._family = np.empty(0, dtype=np.intp)
        self._line = np.empty(0, dtype=np.intp)
        self._position = np.empty(0, dtype=np.float32)
        self._speed = np.empty(0, dtype=np.float32)
        self._tail = np.empty(0, dtype=np.float32)
        self._reverse = np.empty(0, dtype=bool)
        # Quanto cada pulso ja andou: a cauda cresce a partir do ponto onde nasceu.
        self._traveled = np.empty(0, dtype=np.float32)

    def _append(self, family, line, position, speed, tail, reverse) -> None:
        self._family = np.concatenate((self._family, family))
        self._line = np.concatenate((self._line, line))
        self._position = np.concatenate((self._position, np.asarray(position, np.float32)))
        self._speed = np.concatenate((self._speed, np.asarray(speed, np.float32)))
        self._tail = np.concatenate((self._tail, np.asarray(tail, np.float32)))
        self._reverse = np.concatenate((self._reverse, reverse))
        self._traveled = np.concatenate((self._traveled, np.zeros(len(family), np.float32)))

    def spawn_at(self, family: int, line: int, sample: float) -> None:
        """Dois pulsos nascendo na amostra sample da linha, um para cada lado."""
        room = min(2, self._render.pulse_max - len(self))
        if room <= 0:
            return
        samples = float(self._sample_counts[family])
        speed = samples / self._rng.uniform(*PULSE_CROSSING_SECONDS)
        tail = float(np.clip(speed * PULSE_TAIL_SECONDS, *PULSE_TAIL_SAMPLES))
        reverse = np.array((False, True)[:room])
        # Posicao conta da ponta de onde o pulso vem: o de volta comeca em last - sample.
        position = np.where(reverse, samples - 1.0 - sample, sample)
        self._append(
            np.full(room, family, np.intp), np.full(room, line, np.intp), position,
            np.full(room, speed), np.full(room, tail), reverse,
        )

    def _spawn(self, count: int) -> None:
        total = int(self._line_counts.sum())
        count = min(count, self._render.pulse_max - len(self))
        if total == 0 or count <= 0:
            return
        # Linha sorteada entre todas, das duas direcoes do xadrez.
        picks = self._rng.integers(0, total, count)
        family = np.searchsorted(np.cumsum(self._line_counts), picks, side="right")
        line = picks - np.concatenate(([0], np.cumsum(self._line_counts)))[family]
        samples = self._sample_counts[family]
        speed = samples / self._rng.uniform(*PULSE_CROSSING_SECONDS, count)
        tail = np.clip(speed * PULSE_TAIL_SECONDS, *PULSE_TAIL_SAMPLES)
        self._append(
            family, line, np.zeros(count), speed, tail, self._rng.random(count) < 0.5
        )

    def advance(self, dt: float) -> None:
        """Move os pulsos dt segundos, descarta os que sairam e sorteia novos."""
        self._position += self._speed * dt
        self._traveled += self._speed * dt
        alive = self._position - self._tail < self._sample_counts[self._family] - 1
        for name in (
            "_family", "_line", "_position", "_speed", "_tail", "_reverse", "_traveled"
        ):
            setattr(self, name, getattr(self, name)[alive])
        if self.rate > 0 and dt > 0:
            self._spawn(int(self._rng.poisson(self.rate * dt)))

    def draw(
        self, image: np.ndarray, family_curves: list, hue: float, brightness: float = 1.0
    ) -> None:
        if len(self) == 0:
            return
        pure = np.array(plaid_palette(hue)[0], dtype=np.float32)
        core = (pure + PULSE_CORE_TINT * (255.0 - pure)) * brightness
        pure *= brightness
        # Cauda amostrada em pontos fixos da ponta de tras ate a cabeca, todos os
        # pulsos de uma vez; posicoes fora da linha colapsam nas pontas.
        last = self._sample_counts[self._family] - 1.0
        head = np.where(self._reverse, last - self._position, self._position)
        tail = np.minimum(self._tail, self._traveled)
        back = head + np.where(self._reverse, tail, -tail)
        fractions = np.linspace(0.0, 1.0, PULSE_FADE_STEPS * PULSE_STEP_POINTS + 1)
        positions = np.clip(
            back[:, None] + (head - back)[:, None] * fractions, 0.0, last[:, None]
        )
        points = np.empty((*positions.shape, 2), dtype=np.float32)
        for family, curves in enumerate(family_curves):
            selected = self._family == family
            if curves is None or not selected.any():
                continue
            position = positions[selected]
            index = np.minimum(position.astype(np.intp), curves.shape[1] - 2)
            fraction = (position - index)[..., None]
            lines = self._line[selected][:, None]
            points[selected] = (
                curves[lines, index] * (1.0 - fraction) + curves[lines, index + 1] * fraction
            )

        subpixel = float(1 << PULSE_SUBPIXEL_SHIFT)
        core_points = np.rint(points * subpixel).astype(np.int32)
        glow_points = np.rint(points * (subpixel / PULSE_GLOW_DOWNSCALE)).astype(np.int32)
        # Trecho de cada nivel de brilho, do mais apagado (atras) a cabeca.
        steps = [
            (
                (step + 1) / PULSE_FADE_STEPS,
                slice(step * PULSE_STEP_POINTS, (step + 1) * PULSE_STEP_POINTS + 1),
            )
            for step in range(PULSE_FADE_STEPS)
        ]
        self._glow[:] = 0
        for brightness, section in steps:
            cv2.polylines(
                self._glow,
                list(glow_points[:, section]),
                False,
                tuple((pure * brightness).tolist()),
                PULSE_GLOW_THICKNESS,
                self._render.line_type,
                PULSE_SUBPIXEL_SHIFT,
            )
        # Halo: borrado em baixa resolucao, ampliado e somado (satura no branco).
        glow = cv2.GaussianBlur(self._glow, (PULSE_GLOW_BLUR, PULSE_GLOW_BLUR), 0)
        glow = cv2.convertScaleAbs(glow, alpha=PULSE_GLOW_GAIN)
        height, width = image.shape[:2]
        cv2.add(
            image,
            cv2.resize(glow, (width, height), interpolation=cv2.INTER_LINEAR),
            dst=image,
        )
        self._core[:] = 0
        for brightness, section in steps:
            cv2.polylines(
                self._core,
                list(core_points[:, section]),
                False,
                tuple((core * brightness).tolist()),
                PULSE_CORE_THICKNESS,
                self._render.line_type,
                PULSE_SUBPIXEL_SHIFT,
            )
        cv2.add(image, self._core, dst=image)


class PlaidField:
    """Xadrez Madras de linhas retas que se cruzam, movidas pelo optical flow.

    O padrao gira e escala em torno do centro do quadro; as linhas que nao cruzam
    o quadro sao descartadas.
    """

    def __init__(
        self,
        width: int,
        height: int,
        sett: tuple[float, np.ndarray, np.ndarray, np.ndarray],
        phases: tuple[float, float],
        density: float,
        rotation: float,
        pulse_rate: float,
        rng: np.random.Generator,
        brightness: float = 1.0,
        thickness_delta: int = 0,
        render: PlaidRender = PlaidRender(),
    ) -> None:
        self.size = (width, height)
        self.brightness = brightness
        period, offsets, tints, thicknesses = sett
        thicknesses = np.maximum(thicknesses + thickness_delta, 1)
        # O sett esta em fracoes da altura do quadro; densidade maior o encolhe.
        unit = height / density
        period_px = period * unit
        angle = math.radians(rotation)
        direction = np.array((math.cos(angle), math.sin(angle)), dtype=np.float32)
        normal = np.array((-direction[1], direction[0]), dtype=np.float32)
        center = np.array((width / 2.0, height / 2.0), dtype=np.float32)

        def half_extent(axis: np.ndarray) -> float:
            """Metade da projecao do quadro sobre o eixo."""
            return (width * abs(axis[0]) + height * abs(axis[1])) / 2.0

        families = []
        # Mesmo sett nas duas direcoes, como no tecido, cada uma com fase propria.
        for along, across, phase in ((direction, normal, phases[0]), (normal, direction, phases[1])):
            reach = half_extent(across)
            repeats = np.arange(
                math.floor(-reach / period_px + phase) - 1,
                math.ceil(reach / period_px + phase) + 1,
            )
            positions = (
                (repeats[:, None] - phase) * period_px + offsets[None] * unit
            ).ravel()
            visible = np.abs(positions) <= reach
            line_tints = np.tile(tints, len(repeats))[visible]
            line_thicknesses = np.tile(thicknesses, len(repeats))[visible]
            # As pontas ficam PLAID_NODE_SPACING alem da borda do quadro.
            length = half_extent(along) + PLAID_NODE_SPACING
            node_count = max(3, math.ceil(2 * length / PLAID_NODE_SPACING) + 1)
            steps = np.linspace(-length, length, node_count, dtype=np.float32)
            rest = (
                center
                + steps[None, :, None] * along
                + positions[visible, None, None].astype(np.float32) * across
            )
            families.append(
                ElasticLines(rest, line_tints, line_thicknesses, render.segment_samples)
            )
        self._families = tuple(families)
        self.pulses = NeonPulses(
            [len(family) for family in self._families],
            [family.sample_count for family in self._families],
            pulse_rate,
            rng,
            (width, height),
            render,
        )

    def __len__(self) -> int:
        return sum(len(family) for family in self._families)

    def reset(self) -> None:
        for family in self._families:
            family.reset()
        self.pulses.clear()

    def update(
        self, flow: Optional[np.ndarray], scale: np.ndarray, threshold: float, gain: float
    ) -> None:
        for family in self._families:
            family.update(flow, scale, threshold, gain)

    def line_batches(self, hue: float) -> tuple[list, list]:
        """Curvas em float de cada familia e lotes (espessura, cor, polilinhas)."""
        palette = [
            tuple(int(channel * self.brightness) for channel in color)
            for color in plaid_palette(hue)
        ]
        family_curves, batches = [], []
        for family in self._families:
            if len(family) == 0:
                family_curves.append(None)
                continue
            float_curves = family.curves()
            family_curves.append(float_curves)
            curves = np.rint(float_curves).astype(np.int32)
            for tint, thickness, indices in family.groups:
                batches.append((thickness, palette[tint], list(curves[indices])))
        return family_curves, batches


class PlaidStack:
    """Malhas de xadrez sobrepostas, desenhadas da mais distante a mais proxima."""

    def __init__(
        self,
        width: int,
        height: int,
        patterns: list[
            tuple[tuple[float, np.ndarray, np.ndarray, np.ndarray], tuple[float, float], float]
        ],
        layout: str,
        density: float,
        rotation: float,
        pulse_rate: float,
        rng: np.random.Generator,
        render: PlaidRender = PlaidRender(),
    ) -> None:
        self.size = (width, height)
        self._render = render
        # Disposicao, densidade e rotacao fixas destas malhas; mudar uma as recria.
        self.layout = (layout, density, rotation)
        # Ganho e matiz valem no proximo frame, sem recriar as malhas.
        self.gain = 1.0
        self.hue = 0.0
        self._layers = []
        for pattern_index, (density_factor, brightness, push, thickness_delta) in PLAID_LAYOUTS[layout]:
            sett, phases, angle_offset = patterns[pattern_index]
            field = PlaidField(
                width,
                height,
                sett,
                phases,
                density * density_factor,
                rotation + angle_offset,
                0.0,
                rng,
                brightness,
                thickness_delta,
                render,
            )
            self._layers.append((field, push))
        # --pulses e o total: cada camada recebe a parte proporcional as suas linhas.
        total = len(self)
        for field, _ in self._layers:
            field.pulses.rate = pulse_rate * len(field) / total if total else 0.0
        self._scratch = np.zeros((height, width, 3), dtype=np.uint8)
        # Curvas do ultimo desenho, por camada e familia, para achar as linhas que
        # as maos tocam; e o instante do ultimo disparo de cada linha.
        self._last_curves: list[list[Optional[np.ndarray]]] = []
        self._touch_times: dict[tuple[int, int, int], float] = {}
        self._bucket = HAND_LINES_PER_SECOND
        self._bucket_time = 0.0

    def __len__(self) -> int:
        return sum(len(field) for field, _ in self._layers)

    def touch(
        self, segments: list[tuple[np.ndarray, np.ndarray]], radius: float, now: float
    ) -> list[int]:
        """Dispara pulsos na linha mais proxima do caminho de cada mao, a ate radius.

        Cada segmento e o caminho de uma mao entre duas deteccoes; os pulsos nascem
        no ponto da linha mais proximo dele. Devolve os indices dos segmentos que
        dispararam.
        """
        fired = []
        # Balde de fichas: enche HAND_LINES_PER_SECOND por segundo, ate um segundo de folga.
        elapsed = now - self._bucket_time
        self._bucket_time = now
        self._bucket = min(HAND_LINES_PER_SECOND, self._bucket + elapsed * HAND_LINES_PER_SECOND)
        for segment_index, (start, end) in enumerate(segments):
            if self._bucket < 1.0:
                break
            low = np.minimum(start, end) - radius
            high = np.maximum(start, end) + radius
            step = end - start
            length2 = max(float(step @ step), 1e-6)
            candidates = []
            for layer, family_curves in enumerate(self._last_curves):
                for family, curves in enumerate(family_curves):
                    if curves is None:
                        continue
                    inside = np.all((curves >= low) & (curves <= high), axis=2)
                    if not inside.any():
                        continue
                    lines, samples = np.nonzero(inside)
                    points = curves[lines, samples]
                    along = np.clip(((points - start) @ step) / length2, 0.0, 1.0)
                    nearest = start + along[:, None] * step
                    distance = np.hypot(*(points - nearest).T)
                    near = distance <= radius
                    if not near.any():
                        continue
                    lines, samples, distance = lines[near], samples[near], distance[near]
                    # A amostra mais proxima de cada linha tocada.
                    order = np.lexsort((distance, lines))
                    _, first = np.unique(lines[order], return_index=True)
                    for index in order[first]:
                        candidates.append(
                            (float(distance[index]), layer, family, int(lines[index]), float(samples[index]))
                        )
            for _, layer, family, line, sample in sorted(candidates):
                key = (layer, family, line)
                if now - self._touch_times.get(key, -math.inf) < HAND_LINE_COOLDOWN:
                    continue
                self._touch_times[key] = now
                self._layers[layer][0].pulses.spawn_at(family, line, sample)
                self._bucket -= 1.0
                fired.append(segment_index)
                break
        return fired

    @property
    def pulse_count(self) -> int:
        return sum(len(field.pulses) for field, _ in self._layers)

    def reset(self) -> None:
        for field, _ in self._layers:
            field.reset()
        self._touch_times.clear()

    def update(
        self, flow: Optional[np.ndarray], scale: np.ndarray, threshold: float
    ) -> None:
        for field, push in self._layers:
            field.update(flow, scale, threshold, self.gain * push)

    def advance(self, dt: float) -> None:
        for field, _ in self._layers:
            field.pulses.advance(dt)

    def draw(self, image: np.ndarray) -> None:
        drawn = [field.line_batches(self.hue) for field, _ in self._layers]
        self._last_curves = [family_curves for family_curves, _ in drawn]
        if not self._render.antialias:
            # Sem anti-aliasing, linha grossa direto no quadro ja e barata; da camada
            # mais distante a mais proxima, as finas por cima das grossas.
            for _, batches in drawn:
                for thickness, color, lines in sorted(batches, key=lambda batch: -batch[0]):
                    cv2.polylines(image, lines, False, color, thickness, cv2.LINE_8)
            for (field, _), (family_curves, _) in zip(self._layers, drawn):
                field.pulses.draw(image, family_curves, self.hue, field.brightness)
            return
        thick: dict[int, list] = {}
        # Finas direto no quadro, da camada mais distante a mais proxima.
        for _, batches in drawn:
            for thickness, color, lines in batches:
                if thickness == 1:
                    cv2.polylines(image, lines, False, color, 1, cv2.LINE_AA)
                else:
                    thick.setdefault(thickness, []).append((color, lines))
        # Linha anti-aliased grossa custa ~3x a fina no OpenCV; as grossas de todas
        # as camadas sao desenhadas finas numa camada por espessura, engrossadas com
        # dilate e compostas com max (~2,8x mais rapido). No max vence a cor mais
        # clara, que e a da camada mais proxima.
        for thickness, groups in thick.items():
            self._scratch[:] = 0
            for color, lines in groups:
                cv2.polylines(self._scratch, lines, False, color, 1, cv2.LINE_AA)
            kernel = np.ones((thickness, thickness), dtype=np.uint8)
            cv2.max(image, cv2.dilate(self._scratch, kernel), dst=image)
        for (field, _), (family_curves, _) in zip(self._layers, drawn):
            field.pulses.draw(image, family_curves, self.hue, field.brightness)


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    return blue, green, red


class MeadowField:
    """Mato de tracos retos em tres camadas; so a do meio se move com o fluxo.

    Base e frente sao desenhadas uma vez em imagens guardadas: por quadro, custam
    uma copia e uma composicao com mascara. Os tracos do meio sao rigidos e o fluxo
    nas duas pontas os arrasta e gira de vez: a paisagem guarda o que os corpos
    mudaram, ate reset() devolver a original.
    """

    def __init__(
        self, width: int, height: int, density: float, rng: np.random.Generator
    ) -> None:
        self.size = (width, height)
        self.gain = 1.0
        self._rng = rng
        colors = np.array([hex_to_bgr(color) for color, _ in MEADOW_PALETTE], np.float32)
        weights = np.array([weight for _, weight in MEADOW_PALETTE], np.float64)
        self._palette_weights = weights / weights.sum()
        area = width * height / (1280 * 720) * density
        counts = {name: max(1, round(layer[0] * area)) for name, layer in MEADOW_LAYERS.items()}
        self._palettes = {
            name: [tuple(int(c) for c in color * brightness) for color in colors]
            for name, (_, brightness, _) in MEADOW_LAYERS.items()
        }

        self._base = np.zeros((height, width, 3), np.uint8)
        self._draw_strokes(self._base, *self._random_strokes(counts["base"], "base"), "base")
        self._front = np.zeros((height, width, 3), np.uint8)
        self._draw_strokes(self._front, *self._random_strokes(counts["front"], "front"), "front")
        # Mascara da frente sem a franja do anti-aliasing: so pixels com mais da
        # metade da cobertura tampam o que esta atras.
        coverage = np.zeros((height, width), np.uint8)
        centers, angles, half_lengths, _ = self._last_strokes
        cv2.polylines(
            coverage, list(self._segments(centers, angles, half_lengths)), False, 255,
            MEADOW_THICKNESS, cv2.LINE_AA, 2,
        )
        _, self._front_mask = cv2.threshold(coverage, 127, 255, cv2.THRESH_BINARY)

        self._centers, self._angles, self._half_lengths, tints = self._random_strokes(
            counts["middle"], "middle"
        )
        self._groups = [np.flatnonzero(tints == index) for index in range(len(colors))]
        self._shift = np.zeros_like(self._centers)
        self._velocity = np.zeros_like(self._centers)
        self._turn = np.zeros_like(self._angles)
        self._spin = np.zeros_like(self._angles)

    def __len__(self) -> int:
        return len(self._centers)

    def _random_strokes(self, count: int, layer: str):
        width, height = self.size
        centers = self._rng.uniform((0.0, 0.0), (width, height), (count, 2)).astype(np.float32)
        angles = self._rng.uniform(0.0, math.pi, count).astype(np.float32)
        lengths = self._rng.uniform(*MEADOW_LAYERS[layer][2], count)
        half_lengths = (lengths * height / 2.0).astype(np.float32)
        tints = self._rng.choice(len(MEADOW_PALETTE), count, p=self._palette_weights)
        self._last_strokes = (centers, angles, half_lengths, tints)
        return centers, angles, half_lengths, tints

    @staticmethod
    def _segments(centers: np.ndarray, angles: np.ndarray, half_lengths: np.ndarray) -> np.ndarray:
        """Pontas (tracos, 2, 2) em coordenadas subpixel (shift 2) para polylines."""
        reach = np.stack((np.cos(angles), np.sin(angles)), axis=1) * half_lengths[:, None]
        ends = np.stack((centers - reach, centers + reach), axis=1)
        return np.rint(ends * 4.0).astype(np.int32)

    def _draw_strokes(self, image, centers, angles, half_lengths, tints, layer: str) -> None:
        segments = self._segments(centers, angles, half_lengths)
        palette = self._palettes[layer]
        for index, color in enumerate(palette):
            selected = segments[tints == index]
            if len(selected):
                cv2.polylines(
                    image, list(selected), False, color, MEADOW_THICKNESS, cv2.LINE_AA, 2
                )

    def reset(self) -> None:
        """Devolve a paisagem sorteada no inicio, sem as mudancas dos corpos."""
        for state in (self._shift, self._velocity, self._turn, self._spin):
            state[:] = 0.0

    def update(
        self, flow: Optional[np.ndarray], scale: np.ndarray, threshold: float
    ) -> None:
        self._velocity *= MEADOW_DAMPING
        self._spin *= MEADOW_DAMPING
        if flow is not None:
            angles = self._angles + self._turn
            reach = np.stack((np.cos(angles), np.sin(angles)), axis=1) * self._half_lengths[:, None]
            centers = self._centers + self._shift
            ends = np.stack((centers - reach, centers + reach), axis=1)
            push = sample_flow(flow, ends, scale, threshold)
            self._velocity += MEADOW_FLOW_PUSH * self.gain * push.mean(axis=1)
            # Torque: a ponta + recebe o fluxo f1 e a ponta - recebe f0; giro = d x (f1 - f0).
            difference = push[:, 1] - push[:, 0]
            torque = reach[:, 0] * difference[:, 1] - reach[:, 1] * difference[:, 0]
            self._spin += MEADOW_TURN_PUSH * self.gain * torque / np.maximum(
                self._half_lengths * self._half_lengths, 1.0
            )
        self._shift += self._velocity
        self._turn += self._spin
        # Sem limite de deslocamento, so dentro do quadro: o traco nao some na borda.
        width, height = self.size
        centers = self._centers + self._shift
        np.clip(centers[:, 0], 0.0, width, out=centers[:, 0])
        np.clip(centers[:, 1], 0.0, height, out=centers[:, 1])
        self._shift = centers - self._centers

    def draw(self, image: np.ndarray) -> None:
        np.copyto(image, self._base)
        segments = self._segments(
            self._centers + self._shift, self._angles + self._turn, self._half_lengths
        )
        for color, indices in zip(self._palettes["middle"], self._groups):
            if indices.size:
                cv2.polylines(
                    image, list(segments[indices]), False, color, MEADOW_THICKNESS,
                    cv2.LINE_AA, 2,
                )
        cv2.copyTo(self._front, self._front_mask, image)


def list_landscape_images(path: str) -> list[str]:
    """Imagens de uma pasta, em ordem alfabetica, ou o proprio arquivo."""
    if not os.path.isdir(path):
        return [path]
    images = sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.lower().endswith(LANDSCAPE_EXTENSIONS)
    )
    if not images:
        raise RuntimeError(f"Nenhuma imagem ({', '.join(LANDSCAPE_EXTENSIONS)}) em {path}")
    return images


def fit_cover(image: np.ndarray, width: int, height: int, anchor: float) -> np.ndarray:
    """Amplia a imagem ate cobrir width x height sem distorcer e corta o excesso."""
    image_height, image_width = image.shape[:2]
    scale = max(width / image_width, height / image_height)
    size = (max(width, round(image_width * scale)), max(height, round(image_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, size, interpolation=interpolation)
    left = (size[0] - width) // 2
    top = round((size[1] - height) * anchor)
    return np.ascontiguousarray(resized[top : top + height, left : left + width])


class LandscapeCanvas:
    """Imagem que os corpos borram e espalham, arrastando a tinta com o fluxo.

    A cada quadro a tela e reamostrada ao longo do fluxo (adveccao semi-Lagrangiana
    com cv2.remap): cada pixel busca a cor de onde o movimento veio. Onde nada se
    move o mapa e a identidade e a imagem fica intacta; onde ha movimento, as
    cerdas fazem cada pixel andar um pouco diferente e o rastro vira pincelada. As
    mudancas sao permanentes ate reset().
    """

    def __init__(
        self,
        width: int,
        height: int,
        image: np.ndarray,
        anchor: float,
        index: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.size = (width, height)
        self.gain = 1.0
        # Sorteia as cerdas; o do processador, para uma semente fixa valer para tudo.
        self._rng = rng if rng is not None else np.random.default_rng()
        self.regen_delay = LANDSCAPE_REGEN_DELAY
        self.regen_time = LANDSCAPE_REGEN_TIME
        # Segundos sem movimento por celula do fluxo; criado com o primeiro fluxo.
        self._idle: Optional[np.ndarray] = None
        self._regen_elapsed = 0.0
        self._regen_frames = 0
        self.set_image(image, index, anchor)
        self._grid_x, self._grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        self.set_bristles(LANDSCAPE_BRISTLE_WIDTH)
        self.backend = LANDSCAPE_SMEAR_BACKEND

    def __len__(self) -> int:
        return 1

    def set_bristles(self, bristle_width: float) -> None:
        """Refaz as cerdas com feixes de bristle_width px; o borrao ja feito fica."""
        width, height = self.size
        self.bristle_width = bristle_width
        noise = self._rng.standard_normal((height, width)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), bristle_width / 2.0)
        noise /= max(float(noise.std()), 1e-6)
        self._bristles = np.clip(1.0 + LANDSCAPE_BRISTLE * noise, 0.0, 2.0)

    def set_image(self, image: np.ndarray, index: int, anchor: float) -> None:
        """Troca a imagem e recomeca a tela limpa; grade e cerdas continuam."""
        width, height = self.size
        self.index = index
        self.anchor = anchor
        self._original = fit_cover(image, width, height, anchor)
        # Faixa de brilho da pintura, para esticar o gradiente das paletas neon.
        gray = cv2.cvtColor(self._original, cv2.COLOR_BGR2GRAY)
        low, high = np.percentile(gray, LANDSCAPE_PALETTE_STRETCH)
        self.levels = (int(low), int(high))
        self._canvas = self._original.copy()

    def reset(self) -> None:
        """Devolve a imagem original, sem as pinceladas dos corpos."""
        np.copyto(self._canvas, self._original)

    def update(
        self, flow: Optional[np.ndarray], scale: np.ndarray, threshold: float
    ) -> None:
        if flow is None:
            return
        flow = flow.copy()
        still = np.hypot(flow[..., 0], flow[..., 1]) <= threshold
        flow[still] = 0.0
        # Celulas com movimento (e vizinhas, que o borrao alcanca) zeram o relogio.
        moving = cv2.dilate((~still).astype(np.uint8), np.ones((3, 3), np.uint8))
        if self._idle is None or self._idle.shape != still.shape:
            self._idle = np.zeros(still.shape, np.float32)
        self._idle[moving > 0] = 0.0
        if self.gain == 0 or not flow.any():
            return
        flow = cv2.GaussianBlur(flow, (LANDSCAPE_FLOW_BLUR, LANDSCAPE_FLOW_BLUR), 0)
        width, height = self.size
        full = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        # Fluxo do quadro reduzido -> px/quadro do quadro original.
        strength = self.gain * LANDSCAPE_FLOW_SCALE
        steps = (strength / float(scale[0]), strength / float(scale[1]))
        maps = self._maps_numpy if self.backend == "numpy" else self._maps_opencv
        map_x, map_y = maps(full, steps)
        self._canvas = cv2.remap(
            self._canvas, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

    def regenerate(self, dt: float) -> None:
        """Devolve aos poucos a pintura original onde ninguem age ha um tempo."""
        if self._idle is None:
            return
        self._idle += dt
        self._regen_elapsed += dt
        self._regen_frames += 1
        if self._regen_frames < LANDSCAPE_REGEN_EVERY:
            return
        elapsed, self._regen_elapsed, self._regen_frames = self._regen_elapsed, 0.0, 0
        ramp = np.clip((self._idle - self.regen_delay) / LANDSCAPE_REGEN_RAMP, 0.0, 1.0)
        if not ramp.any():
            return
        rate = -math.log(LANDSCAPE_REGEN_RESIDUE) / self.regen_time
        step = 1.0 - math.exp(-rate * elapsed)
        width, height = self.size
        alpha = cv2.resize(
            (ramp * (step * 255.0)).astype(np.uint8), (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        alpha3 = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        # canvas += (original - canvas) * alfa, em 16 bits e com um so arredondamento:
        # misturar os dois termos em 8 bits arredonda duas vezes e trava mais longe.
        delta = cv2.subtract(self._original, self._canvas, dtype=cv2.CV_16S)
        delta = cv2.multiply(delta, alpha3, scale=1.0 / 255.0, dtype=cv2.CV_16S)
        self._canvas = cv2.add(self._canvas, delta, dtype=cv2.CV_8U)
        # Onde a diferenca ja e menor do que o passo consegue mover (|d| * alfa < 0,5),
        # copia o original: sem isso a pintura pararia a alguns niveis dele.
        stuck = math.ceil(0.5 / max(step, 1e-3))
        channels = cv2.split(cv2.absdiff(self._canvas, self._original))
        difference = cv2.max(cv2.max(channels[0], channels[1]), channels[2])
        done = cv2.bitwise_and(
            cv2.inRange(difference, 1, stuck), cv2.inRange(alpha, 1, 255)
        )
        cv2.copyTo(self._original, done, self._canvas)

    def _maps_numpy(
        self, full: np.ndarray, steps: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mapas do remap em NumPy (versao original): grade - fluxo * cerdas * passo."""
        full *= self._bristles[..., None]
        return self._grid_x - full[..., 0] * steps[0], self._grid_y - full[..., 1] * steps[1]

    def _maps_opencv(
        self, full: np.ndarray, steps: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Os mesmos mapas com cv2, sem os arrays intermediarios do NumPy."""
        flow_x, flow_y = cv2.split(full)
        map_x = cv2.scaleAdd(cv2.multiply(flow_x, self._bristles), -steps[0], self._grid_x)
        map_y = cv2.scaleAdd(cv2.multiply(flow_y, self._bristles), -steps[1], self._grid_y)
        return map_x, map_y

    def draw(self, image: np.ndarray) -> None:
        np.copyto(image, self._canvas)


class GrassField:
    """Capim alto de Bezier cubicas finas que o optical flow empurra.

    Posicao, altura e forma de repouso de cada haste sao sorteadas na criacao.
    O unico estado dinamico e o deslocamento horizontal da ponta, uma mola
    amortecida: os controles o seguem pelo perfil de flexao e descem para manter
    o comprimento, como uma haste que tomba presa na base.
    """

    def __init__(
        self,
        width: int,
        height: int,
        density: float,
        max_height: float,
        gain: float,
        rng: np.random.Generator,
    ) -> None:
        count = max(1, round(width * density / 100.0))
        self.size = (width, height)
        self.gain = gain
        self._base = np.stack(
            (
                rng.uniform(0.0, width, count),
                rng.uniform(height * (1.0 - GRASS_BASE_BAND), height, count),
            ),
            axis=1,
        ).astype(np.float32)
        heights = rng.uniform(GRASS_MIN_HEIGHT_RATIO, 1.0, count) * max_height * height
        # Distancia de cada controle a base, usada para descer ao tombar.
        self._lengths = (heights[:, None] * GRASS_CONTROL_LEVELS).astype(np.float32)
        self._max_offset = (GRASS_MAX_BEND * heights).astype(np.float32)
        lean = rng.uniform(-GRASS_REST_LEAN, GRASS_REST_LEAN, (count, 1))
        bend = rng.uniform(-GRASS_REST_BEND, GRASS_REST_BEND, (count, 3))
        bend[:, 2] = 0.0
        # Controles de repouso P1, P2 e ponta, relativos a base: (hastes, 3, 2).
        self._rest = np.stack(
            (heights[:, None] * (lean * GRASS_BEND_PROFILE + bend), -self._lengths),
            axis=2,
        ).astype(np.float32)
        self._offset = np.zeros(count, dtype=np.float32)
        self._velocity = np.zeros(count, dtype=np.float32)
        colors = rng.integers(0, len(GRASS_COLORS), count)
        self._color_groups = [
            (color, np.flatnonzero(colors == index))
            for index, color in enumerate(GRASS_COLORS)
        ]

    def __len__(self) -> int:
        return len(self._base)

    def reset(self) -> None:
        self._offset[:] = 0.0
        self._velocity[:] = 0.0

    def _control_points(self) -> np.ndarray:
        """Posicoes atuais de P1, P2 e ponta: (hastes, 3, 2)."""
        shift = self._offset[:, None] * GRASS_BEND_PROFILE
        drop = self._lengths - np.sqrt(
            np.maximum(self._lengths * self._lengths - shift * shift, 0.0)
        )
        points = self._base[:, None] + self._rest
        points[:, :, 0] += shift
        points[:, :, 1] += drop
        return points

    def update(
        self, flow: Optional[np.ndarray], scale: np.ndarray, threshold: float
    ) -> None:
        """Avanca um frame; flow e o campo reduzido em scale = (sx, sy)."""
        force = 0.0
        if flow is not None:
            push = sample_flow(flow, self._control_points(), scale, threshold)[:, :, 0]
            force = push @ GRASS_FORCE_WEIGHTS
        self._velocity *= GRASS_DAMPING
        self._velocity += GRASS_FLOW_PUSH * self.gain * force - GRASS_STIFFNESS * self._offset
        self._offset += self._velocity
        np.clip(self._offset, -self._max_offset, self._max_offset, out=self._offset)

    def draw(self, image: np.ndarray) -> None:
        anchors = np.concatenate((self._base[:, None], self._control_points()), axis=1)
        curves = np.rint(GRASS_BASIS @ anchors).astype(np.int32)
        for color, indices in self._color_groups:
            if indices.size:
                cv2.polylines(
                    image, list(curves[indices]), False, color, GRASS_THICKNESS, cv2.LINE_AA
                )


class VisionProcessor:
    """Runs one selected 2D algorithm on each new BGR frame."""

    MODE_LABELS = {
        "mog2": "MOG2 (Subtracao de Fundo)",
        "optical_flow": "Optical Flow (Farneback)",
        "conexoes": "YOLOv8n-pose",
        "capim": "Capim (Optical Flow)",
        "mato": "Mato (Optical Flow)",
        "paisagens": "Paisagens (Optical Flow)",
        "xadrez": "Xadrez Madras (Optical Flow)",
    }
    HEAD_NODE = 17
    NODE_GROUPS = (
        (HEAD_NODE, (0, 1, 2, 3, 4)),
        (5, (5, 6)),
        (7, (7, 8)),
        (9, (9, 10)),
        (11, (11, 12)),
        (13, (13, 14)),
        (15, (15, 16)),
    )
    FACE_KEYPOINTS = np.array((0, 1, 2, 3, 4), dtype=np.intp)
    LEFT_KEYPOINTS = np.array((5, 7, 9, 11, 13, 15), dtype=np.intp)
    RIGHT_KEYPOINTS = LEFT_KEYPOINTS + 1
    MOG2_SCALE = 0.25
    FLOW_SCALE = 0.125
    FLOW_THRESHOLD = 0.15
    # Farneback inventa movimento (ate ~3 px/quadro) em areas lisas ou estouradas
    # mesmo entre quadros identicos. So vale o fluxo a ate FLOW_CHANGE_REACH
    # celulas de onde o cinza mudou mais que FLOW_CHANGE_LEVELS niveis.
    FLOW_CHANGE_LEVELS = 2
    FLOW_CHANGE_REACH = 5

    def __init__(
        self,
        mode: str,
        nogpu: bool = False,
        max_distance: float = 150.0,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        point_deadband: float = 4.0,
        show_points: bool = False,
        int8: Optional[bool] = None,
        edge_style: str = "line",
        grass_density: float = DEFAULT_GRASS_DENSITY,
        grass_height: float = DEFAULT_GRASS_HEIGHT,
        grass_gain: float = DEFAULT_GRASS_GAIN,
        meadow_density: float = 1.0,
        meadow_gain: float = 1.0,
        landscape_image: str = LANDSCAPE_DEFAULT_IMAGE,
        landscape_anchor: float = LANDSCAPE_DEFAULT_ANCHOR,
        landscape_regen_delay: float = LANDSCAPE_REGEN_DELAY,
        landscape_regen_time: float = LANDSCAPE_REGEN_TIME,
        landscape_bristle_width: float = LANDSCAPE_BRISTLE_WIDTH,
        landscape_palette: str = "original",
        plaid_density: float = DEFAULT_PLAID_DENSITY,
        plaid_gain: float = DEFAULT_PLAID_GAIN,
        plaid_sensitivity: float = DEFAULT_PLAID_SENSITIVITY,
        plaid_rotation: float = DEFAULT_PLAID_ROTATION,
        plaid_hue: Optional[float] = DEFAULT_PLAID_HUE,
        pulse_rate: float = DEFAULT_PULSE_RATE,
        pulse_source: str = "random",
        silhouette: bool = False,
        plaid_layout: str = "mix",
        render: PlaidRender = PlaidRender(),
    ) -> None:
        if mode not in self.MODE_LABELS:
            raise ValueError(f"Modo 2D desconhecido: {mode}")
        if max_distance <= 0:
            raise ValueError("max_distance deve ser maior que zero")
        if max_connections < 1:
            raise ValueError("max_connections deve ser pelo menos 1")
        if edge_style not in EDGE_STYLES:
            raise ValueError(f"Estilo de aresta desconhecido: {edge_style}")
        if grass_density <= 0:
            raise ValueError("grass_density deve ser maior que zero")
        if not 0 < grass_height <= 1:
            raise ValueError("grass_height deve estar em (0, 1]")
        if grass_gain < 0:
            raise ValueError("grass_gain deve ser >= 0")
        if plaid_density <= 0:
            raise ValueError("plaid_density deve ser maior que zero")
        if plaid_gain < 0:
            raise ValueError("plaid_gain deve ser >= 0")
        if pulse_rate < 0:
            raise ValueError("pulse_rate deve ser >= 0")
        if plaid_layout not in PLAID_LAYOUTS:
            raise ValueError(f"Disposicao de xadrez desconhecida: {plaid_layout}")
        self.plaid_layout = plaid_layout
        self.render = render
        self.pulse_rate = pulse_rate
        if pulse_source not in PULSE_SOURCES:
            raise ValueError(f"Origem de pulsos desconhecida: {pulse_source}")
        self.pulse_source = pulse_source
        # Maos detectadas pelo YOLO (so no xadrez com pulsos das maos).
        self._hand_tracker: Optional[HandTracker] = None
        self._hand_sequence = 0
        self._previous_hands = np.empty((0, 2), dtype=np.float32)
        self._recent_fires: list[tuple[np.ndarray, float]] = []
        # Alternada pela tecla g na thread principal; o worker so le o valor.
        self.silhouette = silhouette
        # Criado na primeira vez que a silhueta precisa do MOG2 (sem Kinect).
        self._silhouette_mog2 = None
        # Quadro de cor solida da silhueta de borda dura, refeito quando cor ou
        # tamanho mudam.
        self._solid: Optional[np.ndarray] = None
        self._solid_key: Optional[tuple] = None
        # Instante do frame anterior do capim/xadrez: os pulsos andam em tempo real.
        # clock pode ser trocado (pi_benchmark.py simula 30 fps sem esperar).
        self._field_clock: Optional[float] = None
        self.clock = time.perf_counter
        self.mode = mode
        self.edge_style = edge_style
        self.grass_density = grass_density
        self.grass_height = grass_height
        self.grass_gain = grass_gain
        if meadow_density <= 0:
            raise ValueError("meadow_density deve ser maior que zero")
        if meadow_gain < 0:
            raise ValueError("meadow_gain deve ser >= 0")
        self.meadow_density = meadow_density
        self.meadow_gain = meadow_gain
        # Forca do borrao do paisagens: comeca no --gain e muda pelas teclas s/x na
        # thread principal; o worker so le o valor a cada quadro.
        self.landscape_gain = min(meadow_gain, LANDSCAPE_MAX_GAIN)
        self.landscape_bristle_width = landscape_bristle_width
        if landscape_palette not in LANDSCAPE_PALETTES:
            raise ValueError(f"Paleta desconhecida: {landscape_palette}")
        self.landscape_palette = landscape_palette
        self._palette_luts: dict[tuple[str, int, int], np.ndarray] = {}
        self.landscape_regen_delay = landscape_regen_delay
        self.landscape_regen_time = landscape_regen_time
        if not 0.0 <= landscape_anchor <= 1.0:
            raise ValueError("landscape_anchor deve estar entre 0 e 1")
        self.landscape_anchor = landscape_anchor
        # Todas as imagens sao lidas ja aqui: um arquivo ruim falha na partida e a
        # troca pelo teclado nao trava esperando o disco.
        self._landscape_images: list[tuple[str, np.ndarray]] = []
        self.landscape_index = 0
        # Pinturas ja ajustadas ao quadro para a janela da silhueta, por (indice, tamanho).
        self._window_cache: dict[tuple[int, int, int], np.ndarray] = {}
        if mode == "paisagens":
            for path in list_landscape_images(landscape_image):
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Nao foi possivel abrir a imagem {path}")
                name = os.path.splitext(os.path.basename(path))[0]
                self._landscape_images.append((name, image))
        # Ajustaveis por teclado (adjust_plaid) enquanto o worker processa: o
        # worker so le os valores e recria a malha quando densidade/rotacao mudam.
        self.plaid_density = min(plaid_density, self._max_plaid_density())
        if mode == "xadrez" and self.plaid_density < plaid_density:
            print(
                f"Aviso: no layout mix a densidade e limitada a {PLAID_MIX_MAX_DENSITY}; "
                f"usando {self.plaid_density} em vez de {plaid_density}.",
                file=sys.stderr,
            )
        self.plaid_gain = plaid_gain
        low, high = PLAID_SENSITIVITY_RANGE
        if not low <= plaid_sensitivity <= high:
            raise ValueError(f"plaid_sensitivity deve estar entre {low} e {high}")
        self.plaid_sensitivity = plaid_sensitivity
        self.plaid_rotation = plaid_rotation % 360.0
        self._rng = np.random.default_rng()
        # Sem matiz (None), o matiz base e sorteado; a CLI comeca no laranja.
        self.plaid_hue = (
            float(self._rng.uniform(0.0, 360.0)) if plaid_hue is None else plaid_hue % 360.0
        )
        # Os padroes do xadrez (sett, fase de cada direcao e desvio de angulo, um
        # por camada do mix) sao sorteados ja aqui, em fracoes da altura do quadro,
        # e mantidos ao mudar densidade, rotacao e disposicao.
        pattern_count = PLAID_PATTERN_COUNT if mode == "xadrez" else 0
        self._plaid_patterns = [
            (make_madras_sett(self._rng), tuple(self._rng.uniform(0.0, 1.0, 2).tolist()), angle)
            for angle in make_plaid_angle_offsets(self._rng, pattern_count)
        ]
        # Capim, mato ou xadrez, criado no primeiro frame, quando o tamanho e conhecido.
        self._field: Optional[GrassField | MeadowField | LandscapeCanvas | PlaidStack] = None
        self.max_distance = max_distance
        self.max_connections = max_connections
        self.show_points = show_points
        self._presentation_deadband = PresentationDeadband(point_deadband)
        # Cor BGR de cada aresta, sorteada quando ela nasce.
        self._mesh_connections: dict[
            tuple[tuple[int, int], tuple[int, int]], tuple[int, int, int]
        ] = {}
        # Pontos de controle Bezier de cada aresta, sorteados quando ela nasce:
        # linhas (t, desvio) para os dois controles, relativas ao segmento a->b.
        self._edge_controls: dict[
            tuple[tuple[int, int], tuple[int, int]], np.ndarray
        ] = {}
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._mog2 = None
        self._previous_gray: Optional[np.ndarray] = None
        self._yolo: Optional[UltralyticsPoseBackend | LetterboxPoseBackend] = None
        # Inferencia que satura a CPU (OpenVINO ou PyTorch CPU).
        self.cpu_inference = False
        self._yolo_backend: Optional[str] = None
        if mode == "mog2":
            self._reset_mog2()
        elif mode == "conexoes":
            self._load_yolo(nogpu, int8)
        elif mode == "xadrez" and pulse_source == "hands":
            self._start_hand_tracker(nogpu, int8)

    def _start_hand_tracker(self, nogpu: bool, int8: Optional[bool]) -> None:
        """Carrega o YOLO-pose para achar as maos; sem ele, volta aos pulsos sorteados."""
        try:
            self._load_yolo(nogpu, int8)
        except Exception as exc:
            print(
                f"Aviso: sem YOLO para achar as maos ({exc}); os pulsos voltam a nascer "
                "sorteados. Para os pulsos das maos, instale requirements.txt.",
                file=sys.stderr,
            )
            self.pulse_source = "random"
            return
        # A inferencia roda na thread do HandTracker, nao na do processamento.
        self.cpu_inference = False
        self._hand_tracker = HandTracker(self._yolo)
        print(f"Pulsos das maos: YOLO-pose em {self._yolo_backend}.")

    def close(self) -> None:
        if self._hand_tracker is not None:
            self._hand_tracker.close()

    @property
    def label(self) -> str:
        if self.mode == "conexoes" and self._yolo_backend:
            return f"YOLOv8 Pose [{self._yolo_backend}]"
        if self.mode == "xadrez":
            return (
                f"Xadrez: dens. {self.plaid_density:.2f} | ganho {self.plaid_gain:.1f} | "
                f"rot. {self.plaid_rotation:.0f} | matiz {self.plaid_hue:.0f} | "
                f"sens. {self.plaid_sensitivity:.2f}"
                + (" | silhueta" if self.silhouette else "")
                + f" | {self.plaid_layout}"
                + (" | pulsos das maos" if self._hand_tracker is not None else "")
            )
        if self.mode == "paisagens":
            name = self._landscape_images[self.landscape_index][0]
            return (
                f"Paisagens: {name} ({self.landscape_index + 1}/{len(self._landscape_images)})"
                f" | forca {self.landscape_gain:.1f}"
                + (
                    f" | recorte {self._field.anchor:.0%}"
                    if isinstance(self._field, LandscapeCanvas)
                    and self._field.index == self.landscape_index
                    else ""
                )
                + (" | silhueta" if self.silhouette else "")
                + f" | paleta {self.landscape_palette}"
                + f" | feixe {self.landscape_bristle_width:.0f}px"
                + f" | volta {self.landscape_regen_delay:.0f}s/{self.landscape_regen_time:.0f}s"
            )
        return self.MODE_LABELS[self.mode]

    def switch_palette(self) -> None:
        """Passa para a proxima paleta (a ultima volta para a original)."""
        names = list(LANDSCAPE_PALETTES)
        self.landscape_palette = names[(names.index(self.landscape_palette) + 1) % len(names)]

    def _palette_lut(self, levels: tuple[int, int]) -> np.ndarray:
        """Tabela (256, 1, 3) BGR da paleta atual, esticada na faixa da pintura."""
        key = (self.landscape_palette, *levels)
        if key not in self._palette_luts:
            stops = np.array(
                [hex_to_bgr(color) for color in LANDSCAPE_PALETTES[self.landscape_palette]],
                np.float32,
            )
            low, high = levels
            brightness = np.clip((np.arange(256) - low) / max(high - low, 1), 0.0, 1.0)
            positions = np.linspace(0.0, 1.0, len(stops))
            self._palette_luts[key] = np.stack(
                [np.interp(brightness, positions, stops[:, channel]) for channel in range(3)],
                axis=1,
            ).astype(np.uint8)[:, None, :]
        return self._palette_luts[key]

    def adjust_bristles(self, width_delta: float) -> None:
        """Afina ou engrossa os feixes do borrao; o worker refaz as cerdas."""
        low, high = LANDSCAPE_BRISTLE_WIDTH_RANGE
        self.landscape_bristle_width = min(high, max(low, self.landscape_bristle_width + width_delta))

    def adjust_regeneration(self, delay_delta: float = 0.0, time_delta: float = 0.0) -> None:
        """Ajusta pelo teclado a espera e o tempo de volta da pintura original."""
        low, high = LANDSCAPE_REGEN_DELAY_RANGE
        self.landscape_regen_delay = min(high, max(low, self.landscape_regen_delay + delay_delta))
        low, high = LANDSCAPE_REGEN_TIME_RANGE
        self.landscape_regen_time = min(high, max(low, self.landscape_regen_time + time_delta))

    def switch_landscape(self, step: int) -> None:
        """Avanca (1) ou volta (-1) na lista de imagens; o worker troca no proximo quadro."""
        self.landscape_index = (self.landscape_index + step) % len(self._landscape_images)

    def adjust_landscape(self, gain_delta: float) -> None:
        """Ajusta a forca do borrao pelo teclado; vale no proximo quadro."""
        self.landscape_gain = round(
            min(LANDSCAPE_MAX_GAIN, max(0.0, self.landscape_gain + gain_delta)), 6
        )

    def toggle_silhouette(self) -> None:
        self.silhouette = not self.silhouette

    def _max_plaid_density(self) -> float:
        high = PLAID_DENSITY_RANGE[1]
        return min(high, PLAID_MIX_MAX_DENSITY) if self.plaid_layout == "mix" else high

    def toggle_layout(self) -> None:
        # A densidade e limitada antes da troca: o worker nunca ve o mix acima do teto.
        if self.plaid_layout == "regular":
            self.plaid_density = min(self.plaid_density, PLAID_MIX_MAX_DENSITY)
            self.plaid_layout = "mix"
        else:
            self.plaid_layout = "regular"

    def adjust_plaid(
        self,
        density_factor: float = 1.0,
        gain_delta: float = 0.0,
        rotation_delta: float = 0.0,
        hue_delta: float = 0.0,
        sensitivity_factor: float = 1.0,
    ) -> None:
        """Ajusta o xadrez pelo teclado; o worker aplica no proximo frame."""
        low = PLAID_DENSITY_RANGE[0]
        self.plaid_density = min(
            self._max_plaid_density(), max(low, self.plaid_density * density_factor)
        )
        self.plaid_gain = round(
            min(PLAID_MAX_GAIN, max(0.0, self.plaid_gain + gain_delta)), 6
        )
        self.plaid_rotation = (self.plaid_rotation + rotation_delta) % 360.0
        self.plaid_hue = (self.plaid_hue + hue_delta) % 360.0
        low, high = PLAID_SENSITIVITY_RANGE
        self.plaid_sensitivity = round(
            min(high, max(low, self.plaid_sensitivity * sensitivity_factor)), 6
        )

    def _load_yolo(self, nogpu: bool, int8: Optional[bool] = None) -> None:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "O modo conexoes (YOLO) requer ultralytics/PyTorch; instale com "
                "python -m pip install -r requirements.txt (em CPU sem placa NVIDIA, veja no "
                "README como instalar o PyTorch so de CPU, bem menor)"
            ) from exc

        if not os.path.isfile(YOLO_PT_MODEL):
            print(
                f"Primeiro uso do modo conexoes: baixando {os.path.basename(YOLO_PT_MODEL)} "
                f"(~7 MB) para {SCRIPT_DIR}; precisa de internet.",
                flush=True,
            )
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available and not nogpu:
            if int8:
                print("Aviso: --int8 so vale para OpenVINO/CPU; use com --nogpu.", file=sys.stderr)
            try:
                self._yolo = CudaGraphPoseBackend(YOLO(YOLO_PT_MODEL))
                self._yolo_backend = "CUDA Graph / NVIDIA GPU"
            except Exception as exc:
                print(
                    f"Aviso: CUDA Graph indisponivel ({exc}); usando Ultralytics em CUDA.",
                    file=sys.stderr,
                )
                self._yolo = UltralyticsPoseBackend(YOLO(YOLO_PT_MODEL), device=0)
                self._yolo_backend = "CUDA / NVIDIA GPU"
            return

        try:
            import openvino  # noqa: F401
        except ImportError:
            self._load_yolo_cpu(YOLO, "OpenVINO ausente")
            return

        # Em CPU, INT8 e o padrao; --no-int8 pede o FP32. Se o INT8 nao puder ser
        # exportado ou carregado, tenta o FP32 antes de cair para PyTorch CPU.
        variants = [(YOLO_OPENVINO_MODEL, False, "OpenVINO / CPU")]
        if int8 is not False:
            variants.insert(0, (YOLO_OPENVINO_INT8_MODEL, True, "OpenVINO INT8 / CPU"))
        failure = ""
        for model_dir, quantized, label in variants:
            if not os.path.isdir(model_dir):
                print(
                    f"Primeiro uso em CPU: exportando o YOLO para OpenVINO"
                    f"{' INT8' if quantized else ''} em {os.path.basename(model_dir)}. Leva "
                    "alguns minutos"
                    + (
                        " e baixa um pequeno dataset de calibracao (precisa de internet)"
                        if quantized
                        else ""
                    )
                    + "; nas proximas execucoes o modelo ja estara pronto.",
                    flush=True,
                )
                try:
                    export_yolo_openvino_model(YOLO, int8=quantized)
                except Exception as exc:
                    failure = f"falha na exportacao OpenVINO: {exc}"
                    print(f"Aviso: {failure}", file=sys.stderr)
                    continue
            try:
                self._yolo = OpenVINOPoseBackend(model_dir)
            except Exception as exc:
                failure = f"falha ao carregar OpenVINO: {exc}"
                print(f"Aviso: {failure}", file=sys.stderr)
                continue
            self._yolo_backend = label
            self.cpu_inference = True
            return
        self._load_yolo_cpu(YOLO, failure)

    def _load_yolo_cpu(self, yolo_class, reason: str) -> None:
        print(
            f"Aviso: {reason}; usando YOLO PyTorch em CPU.",
            file=sys.stderr,
        )
        self._yolo = UltralyticsPoseBackend(yolo_class(YOLO_PT_MODEL), device="cpu")
        self._yolo_backend = "PyTorch CPU"
        self.cpu_inference = True

    def _reset_mog2(self) -> None:
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )

    def reset(self) -> None:
        if self.mode == "mog2":
            self._reset_mog2()
        elif self.mode == "optical_flow":
            self._previous_gray = None
        elif self.mode in ("capim", "mato", "paisagens", "xadrez"):
            # Mantem o capim/xadrez sorteado; um quadro de outro tamanho o recria.
            self._previous_gray = None
            self._field_clock = None
            if self._field is not None:
                self._field.reset()
        elif self.mode == "conexoes":
            self._presentation_deadband.reset()
            self._mesh_connections.clear()
            self._edge_controls.clear()

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask

    @staticmethod
    def _empty_result(frame_bgr: np.ndarray) -> VisionResult:
        height, width = frame_bgr.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        return VisionResult(mask, frame_bgr.copy(), np.zeros_like(frame_bgr))

    def process(
        self, frame_bgr: np.ndarray, body_mask: Optional[np.ndarray] = None
    ) -> VisionResult:
        """body_mask e a mascara de profundidade do Kinect, quando a fonte e Kinect."""
        if self.mode == "mog2":
            height, width = frame_bgr.shape[:2]
            size = (
                max(1, int(width * self.MOG2_SCALE)),
                max(1, int(height * self.MOG2_SCALE)),
            )
            small_bgr = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
            small_mask = self._clean_mask(self._mog2.apply(small_bgr))
            mask = cv2.resize(
                small_mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
            debug = frame_bgr.copy()
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(debug, contours, -1, (0, 255, 0), 1)
            return VisionResult(mask, debug, make_body_visual(mask))

        if self.mode in ("optical_flow", "capim", "mato", "paisagens", "xadrez"):
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(
                gray,
                (
                    max(1, int(gray.shape[1] * self.FLOW_SCALE)),
                    max(1, int(gray.shape[0] * self.FLOW_SCALE)),
                ),
                interpolation=cv2.INTER_AREA,
            )
            if self.mode == "optical_flow":
                result = self._process_flow(frame_bgr, small_gray)
            else:
                result = self._process_field(frame_bgr, small_gray, body_mask)
            self._previous_gray = small_gray
            return result

        return self._process_yolo(frame_bgr)

    def _farneback(self, gray: np.ndarray) -> np.ndarray:
        flow = cv2.calcOpticalFlowFarneback(
            self._previous_gray, gray, None, 0.5, 1, 5, 1, 5, 1.1, 0
        )
        changed = cv2.dilate(
            (cv2.absdiff(self._previous_gray, gray) > self.FLOW_CHANGE_LEVELS).astype(np.uint8),
            np.ones((self.FLOW_CHANGE_REACH, self.FLOW_CHANGE_REACH), np.uint8),
        )
        flow[changed == 0] = 0.0
        return flow

    def _make_field(
        self, width: int, height: int
    ) -> GrassField | MeadowField | LandscapeCanvas | PlaidStack:
        if self.mode == "paisagens":
            index = self.landscape_index
            image = self._landscape_images[index][1]
            self._window_cache.clear()
            return LandscapeCanvas(
                width, height, image, self._crop_anchor(image), index, self._rng
            )
        if self.mode == "mato":
            return MeadowField(width, height, self.meadow_density, self._rng)
        if self.mode == "xadrez":
            return PlaidStack(
                width,
                height,
                self._plaid_patterns,
                self.plaid_layout,
                self.plaid_density,
                self.plaid_rotation,
                # Com as maos, nenhum pulso nasce sorteado: so dos toques.
                0.0 if self._hand_tracker is not None else self.pulse_rate,
                self._rng,
                self.render,
            )
        return GrassField(
            width,
            height,
            self.grass_density,
            self.grass_height,
            self.grass_gain,
            self._rng,
        )

    def _silhouette_mask(
        self, frame_bgr: np.ndarray, body_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Mascara limpa dos corpos (0/255) em SILHOUETTE_SCALE do quadro."""
        height, width = frame_bgr.shape[:2]
        size = (max(1, int(width * SILHOUETTE_SCALE)), max(1, int(height * SILHOUETTE_SCALE)))
        if body_mask is not None:
            small = cv2.resize(body_mask, size, interpolation=cv2.INTER_AREA)
            _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
        else:
            if self._silhouette_mog2 is None:
                self._silhouette_mog2 = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=16, detectShadows=False
                )
            small_bgr = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
            small = self._silhouette_mog2.apply(small_bgr)
        return self._clean_mask(small)

    def _silhouette_fill(
        self, frame_bgr: np.ndarray, body_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Quadro preto com os corpos preenchidos pela cor base do xadrez."""
        height, width = frame_bgr.shape[:2]
        small = self._silhouette_mask(frame_bgr, body_mask)
        blue, green, red = plaid_palette(self.plaid_hue)[0]
        if not self.render.soft_silhouette:
            # Borda dura: amplia so a mascara (1 canal) e recorta um quadro de cor
            # solida, ~40% mais barato que colorir e ampliar 3 canais.
            mask = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
            key = (width, height, blue, green, red)
            if self._solid_key != key:
                self._solid = np.full((height, width, 3), (blue, green, red), np.uint8)
                self._solid_key = key
            return cv2.bitwise_and(self._solid, self._solid, mask=mask)
        small = cv2.GaussianBlur(small, (SILHOUETTE_BLUR, SILHOUETTE_BLUR), 0)
        # Colorir na escala reduzida e depois ampliar da o mesmo resultado (as duas
        # operacoes sao lineares) em ~0,7 ms, contra ~5 ms colorindo em 720p.
        colored = cv2.multiply(
            cv2.cvtColor(small, cv2.COLOR_GRAY2BGR), (blue, green, red, 0), scale=1.0 / 255.0
        )
        return cv2.resize(colored, (width, height), interpolation=cv2.INTER_LINEAR)

    def _landscape_window(self, width: int, height: int) -> np.ndarray:
        """Pintura vista pela silhueta: a proxima da lista, sem borrao.

        Com uma imagem so, e a propria pintura original, sem as pinceladas.
        """
        count = len(self._landscape_images)
        if count == 1:
            return self._field._original
        index = (self.landscape_index + 1) % count
        key = (index, width, height)
        if key not in self._window_cache:
            image = self._landscape_images[index][1]
            self._window_cache[key] = fit_cover(
                image, width, height, self._crop_anchor(image)
            )
        return self._window_cache[key]

    def _crop_anchor(self, image: np.ndarray) -> float:
        """Altura do recorte: sorteada para imagens em retrato, --image-anchor nas outras."""
        height, width = image.shape[:2]
        return float(self._rng.uniform(0.0, 1.0)) if height > width else self.landscape_anchor

    def _silhouette_window(
        self, body: np.ndarray, frame_bgr: np.ndarray, body_mask: Optional[np.ndarray]
    ) -> None:
        """Mostra outra pintura dentro dos corpos, ao vivo, por cima da tela borrada."""
        height, width = body.shape[:2]
        window = self._landscape_window(width, height)
        small = self._silhouette_mask(frame_bgr, body_mask)
        if not self.render.soft_silhouette:
            mask = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
            cv2.copyTo(window, mask, body)
            return
        # Borda macia: alfa ampliado da mascara borrada; ~0,7 ms em 640x480, ~6 ms
        # em 720p (a mistura e em resolucao cheia).
        small = cv2.GaussianBlur(small, (SILHOUETTE_BLUR, SILHOUETTE_BLUR), 0)
        alpha = cv2.cvtColor(
            cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_GRAY2BGR,
        )
        cv2.add(
            cv2.multiply(window, alpha, scale=1.0 / 255.0),
            cv2.multiply(body, 255 - alpha, scale=1.0 / 255.0),
            dst=body,
        )

    def _process_field(
        self, frame_bgr: np.ndarray, gray: np.ndarray, body_mask: Optional[np.ndarray] = None
    ) -> VisionResult:
        """Capim, mato, paisagens e xadrez: desenho empurrado pelo optical flow."""
        height, width = frame_bgr.shape[:2]
        if self._field is None or self._field.size != (width, height):
            self._field = self._make_field(width, height)
            self._previous_gray = None
        elif self.mode == "xadrez" and self._field.layout != (
            self.plaid_layout,
            self.plaid_density,
            self.plaid_rotation,
        ):
            self._field = self._make_field(width, height)
        if self.mode == "xadrez":
            self._field.gain = self.plaid_gain
            self._field.hue = self.plaid_hue
        elif self.mode == "mato":
            self._field.gain = self.meadow_gain
        elif self.mode == "paisagens":
            self._field.gain = self.landscape_gain
            index = self.landscape_index
            if self._field.bristle_width != self.landscape_bristle_width:
                self._field.set_bristles(self.landscape_bristle_width)
            if self._field.index != index:
                image = self._landscape_images[index][1]
                self._field.set_image(image, index, self._crop_anchor(image))
                # A janela da silhueta tambem passa a outra pintura: novo sorteio.
                self._window_cache.clear()
        flow = None if self._previous_gray is None else self._farneback(gray)
        scale = np.array(
            (gray.shape[1] / width, gray.shape[0] / height), dtype=np.float32
        )
        threshold = self.FLOW_THRESHOLD
        if self.mode == "xadrez":
            # Mais sensivel = movimento mais lento ja empurra as linhas.
            threshold /= self.plaid_sensitivity
        self._field.update(flow, scale, threshold)
        if self.mode in ("xadrez", "paisagens"):
            now = self.clock()
            # Limita o passo: uma pausa longa nao deve teletransportar os pulsos nem
            # adiantar a regeneracao.
            dt = 0.0 if self._field_clock is None else min(now - self._field_clock, 0.1)
            self._field_clock = now
            if self.mode == "xadrez":
                self._field.advance(dt)
                if self._hand_tracker is not None:
                    self._touch_lines_with_hands(frame_bgr, now)
            else:
                self._field.regen_delay = self.landscape_regen_delay
                self._field.regen_time = self.landscape_regen_time
                self._field.regenerate(dt)
        # A silhueta fica por baixo; linhas e pulsos sao desenhados por cima.
        if self.mode == "xadrez" and self.silhouette:
            body = self._silhouette_fill(frame_bgr, body_mask)
        else:
            body = np.zeros_like(frame_bgr)
        self._field.draw(body)
        if self.mode == "paisagens" and self.silhouette:
            self._silhouette_window(body, frame_bgr, body_mask)
        if self.mode == "paisagens" and self.landscape_palette != "original":
            body = cv2.applyColorMap(
                cv2.cvtColor(body, cv2.COLOR_BGR2GRAY), self._palette_lut(self._field.levels)
            )
        # Desenhar uma vez e compor com max custa ~0,2 ms; redesenhar as linhas
        # anti-aliased no quadro de debug custaria outros ~3 ms.
        debug = cv2.max(frame_bgr, body) if self.render.debug_output else body
        if self._hand_tracker is not None and self.render.debug_output:
            # As maos so aparecem no debug, nunca na projecao.
            radius = max(4, round(HAND_TOUCH_RADIUS * self.plaid_sensitivity * height))
            for x, y in self._previous_hands.tolist():
                cv2.circle(debug, (round(x), round(y)), radius, (0, 255, 255), 2, cv2.LINE_AA)
        return VisionResult(None, debug, body)

    def _touch_lines_with_hands(self, frame_bgr: np.ndarray, now: float) -> None:
        """Entrega o quadro ao YOLO e dispara pulsos com a deteccao mais recente."""
        self._hand_tracker.submit(frame_bgr)
        sequence, hands = self._hand_tracker.latest()
        if sequence == self._hand_sequence:
            return
        self._hand_sequence = sequence
        height = frame_bgr.shape[0]
        paths = hand_paths(
            self._previous_hands,
            hands,
            HAND_MAX_JUMP * height,
            HAND_MIN_MOVE / self.plaid_sensitivity * height,
        )
        self._previous_hands = hands
        # O YOLO nao da identidade as maos: uma mao que disparou ha menos de
        # HAND_COOLDOWN s e reconhecida pela posicao (perto de onde disparou).
        self._recent_fires = [
            (point, moment) for point, moment in self._recent_fires if now - moment < HAND_COOLDOWN
        ]
        reach = HAND_MAX_JUMP * height
        paths = [
            path
            for path in paths
            if all(np.hypot(*(path[1] - point)) > reach for point, _ in self._recent_fires)
        ]
        if paths:
            radius = HAND_TOUCH_RADIUS * self.plaid_sensitivity * height
            for index in self._field.touch(paths, radius, now):
                self._recent_fires.append((paths[index][1], now))

    def _process_flow(self, frame_bgr: np.ndarray, gray: np.ndarray) -> VisionResult:
        height, width = frame_bgr.shape[:2]
        if self._previous_gray is None:
            return self._empty_result(frame_bgr)
        flow = self._farneback(gray)
        magnitude, _ = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        moving = (magnitude > self.FLOW_THRESHOLD).astype(np.uint8) * 255
        small_mask = self._clean_mask(moving)
        mask = cv2.resize(
            small_mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
        debug = frame_bgr.copy()
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
    def _select_mesh_edges(
        coordinates: np.ndarray,
        max_distance: float,
        max_connections: int,
        degrees: Optional[np.ndarray] = None,
        connected: frozenset[tuple[int, int]] = frozenset(),
        groups: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Greedily links the closest pairs among nodes that still have free slots.

        Nodes that share a value in ``groups`` (the same body) are never linked.
        """
        empty = np.empty(0, dtype=np.intp)
        if degrees is None:
            degrees = np.zeros(len(coordinates), dtype=np.int32)
        else:
            degrees = degrees.astype(np.int32, copy=True)
        free_nodes = np.flatnonzero(degrees < max_connections)
        if free_nodes.size < 2:
            return empty, empty, np.empty(0, dtype=np.float32)

        # Com no maximo algumas centenas de nos, todos os pares vetorizados em NumPy
        # custam menos que qualquer indice espacial em Python.
        pair_a, pair_b = np.triu_indices(free_nodes.size, 1)
        first, second = free_nodes[pair_a], free_nodes[pair_b]
        deltas = coordinates[first] - coordinates[second]
        squared_distances = np.einsum("ij,ij->i", deltas, deltas)
        candidates = squared_distances < max_distance * max_distance
        if groups is not None:
            candidates &= groups[first] != groups[second]
        first, second = first[candidates], second[candidates]
        squared_distances = squared_distances[candidates]
        if first.size == 0:
            return empty, empty, np.empty(0, dtype=np.float32)
        order = np.argsort(squared_distances, kind="stable")
        first_list, second_list = first[order].tolist(), second[order].tolist()
        degree_list = degrees.tolist()
        free_count = free_nodes.size
        selected = []

        for position, (node_a, node_b) in enumerate(zip(first_list, second_list)):
            if degree_list[node_a] >= max_connections or degree_list[node_b] >= max_connections:
                continue
            if (node_a, node_b) in connected:
                continue
            degree_list[node_a] += 1
            degree_list[node_b] += 1
            free_count -= int(degree_list[node_a] == max_connections)
            free_count -= int(degree_list[node_b] == max_connections)
            selected.append(position)
            if free_count < 2:
                break

        selected = order[np.asarray(selected, dtype=np.intp)]
        return (
            first[selected],
            second[selected],
            np.sqrt(squared_distances[selected]),
        )

    def _build_nodes(
        self, points: np.ndarray, confidence: Optional[np.ndarray], width: int, height: int
    ) -> dict[tuple[int, int], tuple[float, float]]:
        """Collapses COCO keypoints into one head node and one node per bilateral class."""
        if points.size == 0:
            return {}
        with np.errstate(invalid="ignore"):
            valid = (
                np.isfinite(points).all(axis=2)
                & (points[:, :, 0] >= 0)
                & (points[:, :, 0] < width)
                & (points[:, :, 1] >= 0)
                & (points[:, :, 1] < height)
            )
        if confidence is not None:
            valid &= confidence[: len(points)] >= YOLO_MIN_KEYPOINT_CONFIDENCE

        face_valid = valid[:, self.FACE_KEYPOINTS]
        face_count = face_valid.sum(axis=1)
        face_points = np.where(face_valid[:, :, None], points[:, self.FACE_KEYPOINTS], 0.0)
        heads = face_points.sum(axis=1) / np.maximum(face_count, 1)[:, None]

        left_valid = valid[:, self.LEFT_KEYPOINTS]
        joints = np.where(
            left_valid[:, :, None],
            points[:, self.LEFT_KEYPOINTS],
            points[:, self.RIGHT_KEYPOINTS],
        )
        joints_valid = left_valid | valid[:, self.RIGHT_KEYPOINTS]

        nodes: dict[tuple[int, int], tuple[float, float]] = {}
        head_list, joint_list = heads.tolist(), joints.tolist()
        for person_index in np.flatnonzero(face_count).tolist():
            nodes[(person_index, self.HEAD_NODE)] = tuple(head_list[person_index])
        for person_index, slot in zip(*np.nonzero(joints_valid)):
            node_class = int(self.LEFT_KEYPOINTS[slot])
            nodes[(int(person_index), node_class)] = tuple(joint_list[person_index][slot])
        return nodes

    def _random_edge_color(self) -> tuple[int, int, int]:
        """Sorteia uma cor qualquer e a leva ao brilho maximo (maior canal = 255)."""
        color = self._rng.uniform(0.0, 1.0, size=3)
        color *= 255.0 / max(color.max(), 1e-6)
        return tuple(int(round(channel)) for channel in color)

    def _random_bezier_controls(self) -> np.ndarray:
        positions = [self._rng.uniform(low, high) for low, high in BEZIER_CONTROL_SPANS]
        bends = self._rng.uniform(-BEZIER_MAX_BEND, BEZIER_MAX_BEND, size=2)
        return np.array(list(zip(positions, bends)), dtype=np.float32)

    def _bezier_polylines(
        self, coordinates: np.ndarray, node_indices: dict[tuple[int, int], int]
    ) -> np.ndarray:
        """Amostra as curvas de todas as arestas, na ordem de _mesh_connections."""
        edges = list(self._mesh_connections)
        if not edges:
            return np.empty((0, BEZIER_SAMPLES, 2), dtype=np.int32)
        start = coordinates[[node_indices[a] for a, _ in edges]]
        end = coordinates[[node_indices[b] for _, b in edges]]
        controls = np.stack([self._edge_controls[edge] for edge in edges])
        direction = end - start
        normal = np.stack((-direction[:, 1], direction[:, 0]), axis=1)
        # Controles acompanham as pontas: escalam e giram junto com a aresta.
        control_points = (
            start[:, None]
            + controls[:, :, :1] * direction[:, None]
            + controls[:, :, 1:] * normal[:, None]
        )
        anchors = np.concatenate(
            (start[:, None], control_points, end[:, None]), axis=1
        )
        curves = np.einsum("sk,ekd->esd", BEZIER_BASIS, anchors)
        return np.rint(curves).astype(np.int32)

    def _process_yolo(self, frame_bgr: np.ndarray) -> VisionResult:
        height, width = frame_bgr.shape[:2]
        points, confidence = yolo_keypoints(self._yolo, frame_bgr)
        raw_nodes = self._build_nodes(points, confidence, width, height)
        stable_nodes = self._presentation_deadband.apply(raw_nodes)
        self._mesh_connections = {
            edge: color
            for edge, color in self._mesh_connections.items()
            if edge[0] in stable_nodes and edge[1] in stable_nodes
        }
        if self.edge_style == "bezier":
            self._edge_controls = {
                edge: self._edge_controls[edge] for edge in self._mesh_connections
            }

        debug = frame_bgr.copy()
        body = np.zeros_like(frame_bgr)
        if not stable_nodes:
            return VisionResult(None, debug, body)

        nodes = list(stable_nodes.items())
        coordinates = np.asarray([position for _, position in nodes], dtype=np.float32)
        node_indices = {key: index for index, (key, _) in enumerate(nodes)}
        degrees = np.zeros(len(nodes), dtype=np.int32)
        connected = set()
        for key_a, key_b in self._mesh_connections:
            index_a, index_b = node_indices[key_a], node_indices[key_b]
            degrees[index_a] += 1
            degrees[index_b] += 1
            connected.add((min(index_a, index_b), max(index_a, index_b)))
        first, second, _ = self._select_mesh_edges(
            coordinates,
            self.max_distance,
            self.max_connections,
            degrees,
            frozenset(connected),
            np.asarray([key[0] for key, _ in nodes], dtype=np.intp),
        )
        for node_a, node_b in zip(first.tolist(), second.tolist()):
            edge = (nodes[node_a][0], nodes[node_b][0])
            self._mesh_connections[edge] = self._random_edge_color()
            if self.edge_style == "bezier":
                self._edge_controls[edge] = self._random_bezier_controls()

        # Desenha direto nos dois quadros: evita addWeighted em tela cheia.
        pixels = np.rint(coordinates).astype(np.int32).tolist()
        curves = (
            self._bezier_polylines(coordinates, node_indices)
            if self.edge_style == "bezier"
            else None
        )
        for edge_index, ((key_a, key_b), color) in enumerate(
            self._mesh_connections.items()
        ):
            if curves is not None:
                curve = [curves[edge_index]]
                cv2.polylines(body, curve, False, color, MESH_THICKNESS, cv2.LINE_AA)
                cv2.polylines(debug, curve, False, color, MESH_THICKNESS, cv2.LINE_AA)
                continue
            point_a = pixels[node_indices[key_a]]
            point_b = pixels[node_indices[key_b]]
            cv2.line(body, point_a, point_b, color, MESH_THICKNESS, cv2.LINE_AA)
            cv2.line(debug, point_a, point_b, color, MESH_THICKNESS, cv2.LINE_AA)

        if self.show_points:
            for (node_key, _), point in zip(nodes, pixels):
                radius = 7 if node_key[1] == self.HEAD_NODE else 5
                for image in (body, debug):
                    cv2.circle(image, point, radius + 3, (0, 55, 0), -1, cv2.LINE_AA)
                    cv2.circle(image, point, radius, (0, 255, 0), -1, cv2.LINE_AA)
        return VisionResult(None, debug, body)


def metronome_click(frequency: float, amplitude: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(METRONOME_CLICK_SECONDS * sample_rate)) / sample_rate
    # Decai ~60 dB ao longo do clique; o ataque curto evita estalo no inicio.
    envelope = np.exp(-t * (6.9 / METRONOME_CLICK_SECONDS)) * np.minimum(1.0, t * sample_rate / 48)
    return (amplitude * envelope * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class Metronome:
    """Metronomo 4/4 sintetizado no callback de audio (sounddevice/PortAudio).

    Cada clique e posicionado na amostra exata dentro do bloco de audio, entao o
    andamento nao depende do FPS do video nem do agendamento das threads Python;
    so ligar, desligar e mudar o andamento esperam o proximo bloco (~20-40 ms).
    """

    def __init__(
        self,
        bpm: float = METRONOME_DEFAULT_BPM,
        enabled: bool = False,
        sample_rate: int = METRONOME_SAMPLE_RATE,
    ) -> None:
        self.sample_rate = sample_rate
        self.bpm = min(max(bpm, METRONOME_BPM_RANGE[0]), METRONOME_BPM_RANGE[1])
        self.enabled = False
        self.error: Optional[str] = None
        self._accent = metronome_click(*METRONOME_ACCENT, sample_rate)
        self._beat_click = metronome_click(*METRONOME_BEAT, sample_rate)
        # Estado do audio, so tocado sob a trava (callback x teclado).
        self._lock = threading.Lock()
        self._sample = 0
        self._next_beat = 0.0
        self._last_beat = 0.0
        self._beat = 0
        self._active: list[tuple[np.ndarray, int]] = []
        self._stream = None
        if enabled:
            self.toggle()

    @property
    def label(self) -> str:
        if self.error is not None:
            return f"Metronomo: sem audio, {self.bpm:.0f} bpm"
        state = "ligado" if self.enabled else "desligado"
        return f"Metronomo: {state}, {self.bpm:.0f} bpm"

    def _interval(self) -> float:
        return self.sample_rate * 60.0 / self.bpm

    def _open_stream(self) -> bool:
        if self._stream is not None:
            return True
        try:
            import sounddevice

            self._stream = sounddevice.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
                # Bloco maior tolera as threads de video segurando o GIL; o tempo
                # dos cliques continua exato.
                latency="high",
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            self.error = f"{type(exc).__name__}: {exc}"
            print(
                f"Aviso: metronomo sem audio ({self.error}). Instale a PortAudio "
                "(sudo apt-get install libportaudio2) e o pacote sounddevice.",
                file=sys.stderr,
            )
            return False
        return True

    def toggle(self) -> None:
        if not self.enabled and not self._open_stream():
            return
        with self._lock:
            self.enabled = not self.enabled
            # Ligar recomeca o compasso pela cabeca, ja no proximo bloco.
            self._beat = 0
            self._next_beat = float(self._sample)
            self._active.clear()

    def adjust(self, steps: float) -> None:
        with self._lock:
            low, high = METRONOME_BPM_RANGE
            self.bpm = min(high, max(low, self.bpm + steps * METRONOME_BPM_STEP))
            if self.enabled and self._beat > 0:
                # O proximo tempo ja segue o novo andamento, contado do ultimo.
                self._next_beat = max(float(self._sample), self._last_beat + self._interval())

    def render(self, frames: int) -> np.ndarray:
        """Proximas frames amostras do metronomo (mono, float32)."""
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            start = self._sample
            end = start + frames
            self._sample = end
            if not self.enabled:
                return out
            while self._next_beat < end:
                click = self._accent if self._beat % METRONOME_BEATS_PER_BAR == 0 else self._beat_click
                self._active.append((click, int(round(self._next_beat))))
                self._last_beat = self._next_beat
                self._beat += 1
                self._next_beat += self._interval()
            still_playing = []
            for click, begin in self._active:
                lo, hi = max(begin, start), min(begin + len(click), end)
                if hi > lo:
                    out[lo - start : hi - start] += click[lo - begin : hi - begin]
                if begin + len(click) > end:
                    still_playing.append((click, begin))
            self._active = still_playing
        return out

    def _callback(self, outdata, frames, time_info, status) -> None:
        outdata[:, 0] = self.render(frames)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


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
    metronome_label: Optional[str] = None,
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
    if metronome_label is not None:
        lines.append(metronome_label)
    text_width = max(
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for text in lines
    )
    box_bottom = 27 + 21 * len(lines)
    cv2.rectangle(debug, (8, 8), (max(630, text_width + 24), box_bottom), (0, 0, 0), -1)
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


def calibrate_kinect_background(capturer: KinectV1Capturer) -> None:
    print("Capture o fundo com a sala vazia; amostrando 15 frames...")
    try:
        capturer.capture_background()
    except RuntimeError as exc:
        print(f"Falha ao calibrar fundo: {exc}", file=sys.stderr)
    else:
        print("Fundo calibrado; apenas objetos mais proximos entrarao na mascara.")


def analyze_frame(
    active_mode: str,
    processor: Optional[VisionProcessor],
    frame_bgr: np.ndarray,
    body_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the (debug, body) images for one captured frame."""
    if active_mode != "kinect":
        result = processor.process(frame_bgr, body_mask)
        return result.debug_frame, result.body_frame
    tinted = np.zeros_like(frame_bgr)
    tinted[:, :, 1] = 200
    overlay = cv2.bitwise_and(tinted, tinted, mask=body_mask)
    debug = cv2.addWeighted(frame_bgr, 1.0, overlay, 0.28, 0.0)
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug, contours, -1, (40, 230, 255), 1)
    return debug, make_body_visual(body_mask)


class ProcessingWorker:
    """Processes the newest captured frame while the main thread drives the windows.

    The Qt event loop (imshow/waitKey) costs a few ms per frame; running vision in
    its own thread overlaps that with the next frame's inference.
    """

    def __init__(self, capturer: CaptureSource, analyze) -> None:
        self._capturer = capturer
        self._analyze = analyze
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._result: Optional[tuple[float, np.ndarray, np.ndarray]] = None
        self._result_ready = threading.Event()
        self.error: Optional[Exception] = None
        self._thread = threading.Thread(
            target=self._loop, name="vision-processing", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        last_timestamp: Optional[float] = None
        try:
            while not self._stop_event.is_set():
                success, frame_bgr, _, body_mask = self._capturer.read()
                timestamp = self._capturer.last_read_timestamp
                if not success or frame_bgr is None or timestamp == last_timestamp:
                    self._stop_event.wait(0.001)
                    continue
                last_timestamp = timestamp
                debug, body = self._analyze(frame_bgr, body_mask)
                with self._lock:
                    self._result = (timestamp, debug, body)
                self._result_ready.set()
        except Exception as exc:
            self.error = exc
            self._result_ready.set()

    def latest(
        self, timeout: float = 0.0
    ) -> Optional[tuple[float, np.ndarray, np.ndarray]]:
        """Waits up to timeout for a new result; returns (timestamp, debug, body)."""
        self._result_ready.wait(timeout)
        self._result_ready.clear()
        with self._lock:
            return self._result

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            # Daemon: nao impede o encerramento, e a captura ainda precisa ser fechada.
            print(
                "Aviso: a thread de processamento nao encerrou em 5 segundos.",
                file=sys.stderr,
            )


class InlineProcessing:
    """ProcessingWorker interface that processes in the calling thread.

    Used for CPU inference: overlapping the Qt repaint with OpenVINO on a saturated
    CPU slowed each inference by ~6 ms (+8 ms latency) without raising the FPS.
    """

    error: Optional[Exception] = None

    def __init__(self, capturer: CaptureSource, analyze) -> None:
        self._capturer = capturer
        self._analyze = analyze
        self._last_timestamp: Optional[float] = None
        self._result: Optional[tuple[float, np.ndarray, np.ndarray]] = None

    def latest(
        self, timeout: float = 0.0
    ) -> Optional[tuple[float, np.ndarray, np.ndarray]]:
        success, frame_bgr, _, body_mask = self._capturer.read()
        timestamp = self._capturer.last_read_timestamp
        if success and frame_bgr is not None and timestamp != self._last_timestamp:
            self._last_timestamp = timestamp
            debug, body = self._analyze(frame_bgr, body_mask)
            self._result = (timestamp, debug, body)
        return self._result

    def close(self) -> None:
        return


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
        choices=("kinect", "mog2", "optical_flow", "conexoes", "capim", "mato", "paisagens", "xadrez"),
        default=None,
        help="algoritmo 2D: kinect, mog2, optical_flow, conexoes (malha YOLO), capim "
        "(capim movido pelo optical flow), mato (tracos em camadas, o do meio movido pelo "
        "optical flow), paisagens (imagem borrada pelo optical flow) ou xadrez (xadrez "
        "Madras movido pelo optical flow) (padrao: xadrez)",
    )
    parser.add_argument(
        "--nogpu",
        action="store_true",
        help="forca YOLO em OpenVINO/CPU ou PyTorch CPU em vez de CUDA",
    )
    parser.add_argument(
        "--int8",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="com OpenVINO/CPU, usa o modelo quantizado INT8 (padrao); --no-int8 usa o FP32: "
        "~40%% mais lento, keypoints mais estaveis",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=150.0,
        help="distancia maxima de conexao entre nos YOLO, em pixels (padrao: 150)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=DEFAULT_MAX_CONNECTIONS,
        help="maximo de ligacoes incidentes por no YOLO (padrao: 5)",
    )
    parser.add_argument(
        "--point-deadband",
        type=float,
        default=4.0,
        help="limiar espacial para ignorar tremor dos nos YOLO, em pixels (padrao: 4)",
    )
    parser.add_argument(
        "--show-points",
        action="store_true",
        help="mostra circulos das articulacoes e cabeca na malha YOLO",
    )
    parser.add_argument(
        "--edge-style",
        choices=EDGE_STYLES,
        default="line",
        help="formato das ligacoes YOLO: line (hastes retas, padrao) ou bezier "
        "(curvas com pontos de controle sorteados quando a ligacao nasce)",
    )
    parser.add_argument(
        "--grass-density",
        type=float,
        default=DEFAULT_GRASS_DENSITY,
        help="capim: hastes de capim por 100 px de largura do quadro (padrao: %(default)s)",
    )
    parser.add_argument(
        "--grass-height",
        type=float,
        default=DEFAULT_GRASS_HEIGHT,
        help="capim: altura maxima das hastes, fracao da altura do quadro; a menor "
        "tem metade (padrao: %(default)s)",
    )
    parser.add_argument(
        "--grass-gain",
        type=float,
        default=DEFAULT_GRASS_GAIN,
        help="capim: quanto o movimento do optical flow empurra o capim (padrao: %(default)s)",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=DEFAULT_PLAID_DENSITY,
        help="xadrez: densidade do padrao Madras; maior = mais linhas e quadros menores "
        "(teclas a/z); mato: multiplica o numero de tracos (padrao: %(default)s)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=DEFAULT_PLAID_GAIN,
        help="xadrez, mato e paisagens: quanto o movimento do optical flow empurra o desenho; "
        "no paisagens e a forca do borrao "
        "(padrao: %(default)s; teclas s/x no xadrez e no paisagens)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=DEFAULT_PLAID_SENSITIVITY,
        help="xadrez: sensibilidade da interacao dos corpos com as linhas: maior = "
        "movimentos menores e mais lentos ja empurram as linhas e as maos disparam "
        "pulsos de mais longe; nao muda a forca do empurrao (--gain) "
        "(padrao: %(default)s, de 0.25 a 4; teclas h/n)",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=DEFAULT_PLAID_ROTATION,
        help="xadrez: rotacao do padrao em graus, em torno do centro do quadro "
        "(padrao: %(default)s; teclas d/c)",
    )
    parser.add_argument(
        "--hue",
        type=float,
        default=DEFAULT_PLAID_HUE,
        help="xadrez: matiz base das linhas em graus (0 vermelho, 120 verde, 240 azul); "
        "as faixas variam so em luminosidade, da cor pura ao branco "
        "(padrao: %(default)s, laranja; teclas f/v giram o disco de cores)",
    )
    parser.add_argument(
        "--pulses",
        type=float,
        default=DEFAULT_PULSE_RATE,
        help="xadrez com --pulse-source random: pulsos de neon que nascem por segundo, "
        "em media, em intervalos aleatorios; 0 desliga (padrao: %(default)s)",
    )
    parser.add_argument(
        "--pulse-source",
        choices=PULSE_SOURCES,
        default="hands",
        help="xadrez: de onde nascem os pulsos: hands (onde as maos tocam as linhas, "
        "achadas pelo YOLO-pose; sem ele, cai para random) ou random (sorteados) "
        "(padrao: %(default)s)",
    )
    parser.add_argument(
        "--silhouette",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="silhueta dos corpos (mascara de profundidade no Kinect, MOG2 nas outras "
        "fontes; tecla g liga/desliga): no xadrez e preenchida pela cor base; no "
        "paisagens e uma janela para a proxima pintura da lista (padrao: ligada no "
        "xadrez, desligada nos outros modos)",
    )
    parser.add_argument(
        "--layout",
        choices=tuple(PLAID_LAYOUTS),
        default="mix",
        help="xadrez: mix (malhas independentes em alturas diferentes, com densidades, "
        f"angulos e paralaxe proprios; densidade ate {PLAID_MIX_MAX_DENSITY}) ou regular "
        "(uma malha) (padrao: mix; tecla l alterna)",
    )
    parser.add_argument(
        "--pi",
        action="store_true",
        help="perfil Raspberry Pi 4: modo xadrez (padrao) ou paisagens; so a janela do "
        "projetor (FPS no terminal); no xadrez, sem anti-aliasing, curvas e silhueta "
        "simplificadas e no maximo 40 pulsos",
    )
    parser.add_argument(
        "--fullscreen",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="abre a janela do projetor em tela cheia (padrao: so com --pi)",
    )
    parser.add_argument(
        "--image",
        default=LANDSCAPE_DEFAULT_IMAGE,
        metavar="ARQUIVO_OU_PASTA",
        help="paisagens: imagem ou pasta de imagens que cobrem a tela; com uma pasta, "
        "a/z alternam entre elas em ordem alfabetica (padrao: a pasta paisagens/)",
    )
    parser.add_argument(
        "--image-anchor",
        type=float,
        default=LANDSCAPE_DEFAULT_ANCHOR,
        help="paisagens: faixa visivel das imagens em paisagem quando sao cortadas, de 0 "
        "(topo) a 1 (base) (padrao: %(default)s, o meio); nas em retrato a altura do "
        "recorte e sorteada a cada carga",
    )
    parser.add_argument(
        "--palette",
        choices=tuple(LANDSCAPE_PALETTES),
        default="original",
        help="paisagens: cores da pintura; original ou uma paleta neon de estetica "
        "cyberpunk aplicada por mapa de gradiente (padrao: original; tecla p alterna)",
    )
    parser.add_argument(
        "--bristle-width",
        type=float,
        default=LANDSCAPE_BRISTLE_WIDTH,
        help="paisagens: espessura dos feixes do borrao, em px: 1-2 estrias finas, ~6 "
        "pinceladas largas, ~12 manchas quase lisas (padrao: %(default)s; teclas h/n)",
    )
    parser.add_argument(
        "--regen-delay",
        type=float,
        default=LANDSCAPE_REGEN_DELAY,
        help="paisagens: segundos sem acao antes de a pintura original comecar a voltar "
        "onde ninguem se move (padrao: %(default)s; teclas d/c)",
    )
    parser.add_argument(
        "--regen-time",
        type=float,
        default=LANDSCAPE_REGEN_TIME,
        help="paisagens: segundos para a pintura voltar ao original depois da espera "
        "(padrao: %(default)s; teclas f/v)",
    )
    parser.add_argument(
        "--metronome",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="toca um metronomo 4/4 com a cabeca do compasso destacada (todos os "
        "modos; tecla t liga/desliga)",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=METRONOME_DEFAULT_BPM,
        help=f"andamento do metronomo, entre {METRONOME_BPM_RANGE[0]:.0f} e "
        f"{METRONOME_BPM_RANGE[1]:.0f} (padrao: %(default)s; teclas - e = ou +)",
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
    if not LANDSCAPE_BRISTLE_WIDTH_RANGE[0] <= args.bristle_width <= LANDSCAPE_BRISTLE_WIDTH_RANGE[1]:
        parser.error("--bristle-width deve estar entre %g e %g" % LANDSCAPE_BRISTLE_WIDTH_RANGE)
    if not LANDSCAPE_REGEN_DELAY_RANGE[0] <= args.regen_delay <= LANDSCAPE_REGEN_DELAY_RANGE[1]:
        parser.error("--regen-delay deve estar entre %g e %g" % LANDSCAPE_REGEN_DELAY_RANGE)
    if not LANDSCAPE_REGEN_TIME_RANGE[0] <= args.regen_time <= LANDSCAPE_REGEN_TIME_RANGE[1]:
        parser.error("--regen-time deve estar entre %g e %g" % LANDSCAPE_REGEN_TIME_RANGE)
    if not 0.0 <= args.image_anchor <= 1.0:
        parser.error("--image-anchor deve estar entre 0 e 1")
    if not METRONOME_BPM_RANGE[0] <= args.bpm <= METRONOME_BPM_RANGE[1]:
        parser.error(
            f"--bpm deve estar entre {METRONOME_BPM_RANGE[0]:.0f} e {METRONOME_BPM_RANGE[1]:.0f}"
        )
    if args.pi:
        if args.mode not in (None, *PI_MODES):
            parser.error(f"--pi so suporta --mode {' ou '.join(PI_MODES)}")
        args.mode = args.mode or PI_MODES[0]
    if args.mode is None:
        args.mode = "xadrez"
    if args.silhouette is None:
        # A silhueta vem ligada so no xadrez; no paisagens ela abriria a janela
        # para outra pintura, que fica como opcao.
        args.silhouette = args.mode == "xadrez"
    if args.mode == "kinect" and args.source != "kinect":
        parser.error("--mode kinect requer --source kinect")
    if args.max_distance <= 0:
        parser.error("--max-distance deve ser maior que zero")
    if args.max_connections < 1:
        parser.error("--max-connections deve ser pelo menos 1")
    if args.point_deadband < 0:
        parser.error("--point-deadband deve ser >= 0")
    if args.grass_density <= 0:
        parser.error("--grass-density deve ser maior que zero")
    if not 0 < args.grass_height <= 1:
        parser.error("--grass-height deve estar entre 0 (exclusivo) e 1")
    if args.grass_gain < 0:
        parser.error("--grass-gain deve ser >= 0")
    if args.density <= 0:
        parser.error("--density deve ser maior que zero")
    if args.gain < 0:
        parser.error("--gain deve ser >= 0")
    if not PLAID_SENSITIVITY_RANGE[0] <= args.sensitivity <= PLAID_SENSITIVITY_RANGE[1]:
        parser.error("--sensitivity deve estar entre %g e %g" % PLAID_SENSITIVITY_RANGE)
    if args.pulses < 0:
        parser.error("--pulses deve ser >= 0")
    return args


def resolve_active_mode(requested_mode: str, capturer: CaptureSource) -> str:
    if requested_mode == "kinect" and not isinstance(capturer, KinectV1Capturer):
        print("Kinect indisponivel; usando MOG2 no modo de fallback.", file=sys.stderr)
        return "mog2"
    return requested_mode


def make_vision_processor(
    mode: str,
    nogpu: bool = False,
    max_distance: float = 150.0,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    point_deadband: float = 4.0,
    show_points: bool = False,
    int8: Optional[bool] = None,
    edge_style: str = "line",
    grass_density: float = DEFAULT_GRASS_DENSITY,
    grass_height: float = DEFAULT_GRASS_HEIGHT,
    grass_gain: float = DEFAULT_GRASS_GAIN,
    meadow_density: float = 1.0,
    meadow_gain: float = 1.0,
    landscape_image: str = LANDSCAPE_DEFAULT_IMAGE,
    landscape_anchor: float = LANDSCAPE_DEFAULT_ANCHOR,
    landscape_regen_delay: float = LANDSCAPE_REGEN_DELAY,
    landscape_regen_time: float = LANDSCAPE_REGEN_TIME,
    landscape_bristle_width: float = LANDSCAPE_BRISTLE_WIDTH,
    landscape_palette: str = "original",
    plaid_density: float = DEFAULT_PLAID_DENSITY,
    plaid_gain: float = DEFAULT_PLAID_GAIN,
    plaid_sensitivity: float = DEFAULT_PLAID_SENSITIVITY,
    plaid_rotation: float = DEFAULT_PLAID_ROTATION,
    plaid_hue: Optional[float] = DEFAULT_PLAID_HUE,
    pulse_rate: float = DEFAULT_PULSE_RATE,
    pulse_source: str = "random",
    silhouette: bool = False,
    plaid_layout: str = "mix",
    render: PlaidRender = PlaidRender(),
) -> Optional[VisionProcessor]:
    return (
        None
        if mode == "kinect"
        else VisionProcessor(
            mode,
            nogpu=nogpu,
            max_distance=max_distance,
            max_connections=max_connections,
            point_deadband=point_deadband,
            show_points=show_points,
            int8=int8,
            edge_style=edge_style,
            grass_density=grass_density,
            grass_height=grass_height,
            grass_gain=grass_gain,
            meadow_density=meadow_density,
            meadow_gain=meadow_gain,
            landscape_image=landscape_image,
            landscape_anchor=landscape_anchor,
            landscape_regen_delay=landscape_regen_delay,
            landscape_regen_time=landscape_regen_time,
            landscape_bristle_width=landscape_bristle_width,
            landscape_palette=landscape_palette,
            plaid_density=plaid_density,
            plaid_gain=plaid_gain,
            plaid_sensitivity=plaid_sensitivity,
            plaid_rotation=plaid_rotation,
            plaid_hue=plaid_hue,
            pulse_rate=pulse_rate,
            pulse_source=pulse_source,
            silhouette=silhouette,
            plaid_layout=plaid_layout,
            render=render,
        )
    )


# Teclas por modo, impressas na partida; q, m e as do metronomo valem sempre.
COMMON_KEY_HELP = "q sai | m troca a fonte | t metronomo, - / = andamento"
MODE_KEY_HELP = {
    "kinect": "a/z profundidade minima | s/x maxima | b calibra o fundo (sala vazia)",
    "xadrez": "a/z densidade | s/x ganho | h/n sensibilidade | d/c rotacao | f/v matiz | g silhueta | "
    "l malha mix/regular | b fundo (Kinect)",
    "paisagens": "a/z pintura | p paleta | s/x forca | h/n feixe | d/c espera da volta | "
    "f/v tempo da volta | g silhueta | b fundo (Kinect)",
}


def print_key_help(mode: str) -> None:
    print(f"Teclas (com a janela em foco): {COMMON_KEY_HELP}")
    if mode in MODE_KEY_HELP:
        print(f"Teclas do modo {mode}: {MODE_KEY_HELP[mode]}")


def display_problem() -> Optional[str]:
    """Motivo pelo qual as janelas do OpenCV nao vao abrir, ou None."""
    # O OpenCV do pip abre janelas por Qt/X11 (xcb); no Wayland, pelo XWayland,
    # que tambem define DISPLAY. Sem ele, o Qt aborta o processo com core dump.
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return (
            "Nenhuma tela grafica disponivel (variavel DISPLAY vazia). Rode a PoC no "
            "terminal da propria sessao grafica do notebook/Raspberry Pi. Por SSH, com a "
            "sessao grafica aberta e logada na maquina, use: export DISPLAY=:0"
        )
    return None


def main() -> int:
    args = parse_args()
    problem = display_problem()
    if problem is not None:
        print(f"Erro: {problem}", file=sys.stderr)
        return 1
    try:
        capturer = create_capturer(
            args.source, args.min_depth, args.max_depth, args.fallback_video
        )
    except RuntimeError as exc:
        print(f"Erro ao iniciar captura: {exc}", file=sys.stderr)
        return 1
    active_mode = resolve_active_mode(args.mode, capturer)
    render = PI_PLAID_RENDER if args.pi else PlaidRender()
    metronome = Metronome(args.bpm, args.metronome)
    try:
        processor = make_vision_processor(
            active_mode,
            nogpu=args.nogpu,
            max_distance=args.max_distance,
            max_connections=args.max_connections,
            point_deadband=args.point_deadband,
            show_points=args.show_points,
            int8=args.int8,
            edge_style=args.edge_style,
            grass_density=args.grass_density,
            grass_height=args.grass_height,
            grass_gain=args.grass_gain,
            meadow_density=args.density,
            meadow_gain=args.gain,
            landscape_image=args.image,
            landscape_anchor=args.image_anchor,
            landscape_regen_delay=args.regen_delay,
            landscape_regen_time=args.regen_time,
            landscape_bristle_width=args.bristle_width,
            landscape_palette=args.palette,
            plaid_density=args.density,
            plaid_gain=args.gain,
            plaid_sensitivity=args.sensitivity,
            plaid_rotation=args.rotation,
            plaid_hue=args.hue,
            pulse_rate=args.pulses,
            pulse_source=args.pulse_source,
            silhouette=args.silhouette,
            plaid_layout=args.layout,
            render=render,
        )
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
    projector_window = "Projetor 2 - Corpo/Frontal"
    # No --pi so a janela do projetor: a de debug custaria outro imshow e o overlay.
    windows = (projector_window,) if args.pi else ("Debug & Tracking", projector_window)
    for window in windows:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # Sem --fullscreen explicito, so o perfil --pi (que roda no projetor) abre em tela cheia.
    fullscreen = args.pi if args.fullscreen is None else args.fullscreen
    cv2.setWindowProperty(
        projector_window,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
    )
    last_status_report = time.perf_counter()

    recent_frame_times = deque(maxlen=30)
    last_latency_ms = 0.0
    last_processed_timestamp: Optional[float] = None

    def start_worker() -> ProcessingWorker | InlineProcessing:
        worker_class = (
            InlineProcessing
            if processor is not None and processor.cpu_inference
            else ProcessingWorker
        )
        return worker_class(
            capturer, functools.partial(analyze_frame, active_mode, processor)
        )

    print_key_help(active_mode)
    worker: Optional[ProcessingWorker | InlineProcessing] = start_worker()
    try:
        while True:
            if capturer.error is not None:
                raise RuntimeError(f"Falha no backend de captura: {capturer.error}")
            if worker.error is not None:
                raise RuntimeError(f"Falha no processamento: {worker.error}") from worker.error
            # Acorda assim que o worker publica, sem esperar o proximo waitKey.
            latest = worker.latest(timeout=0.005)
            is_new_frame = latest is not None and latest[0] != last_processed_timestamp
            if is_new_frame:
                frame_start, debug, body = latest
                recent_frame_times.append(frame_start)
                fps = (
                    (len(recent_frame_times) - 1)
                    / (recent_frame_times[-1] - recent_frame_times[0])
                    if len(recent_frame_times) > 1
                    else 0.0
                )
                if args.pi:
                    cv2.imshow(projector_window, body)
                    now = time.perf_counter()
                    if now - last_status_report >= PI_STATUS_INTERVAL:
                        last_status_report = now
                        print(
                            f"FPS {fps:.1f} | latencia {last_latency_ms:.0f} ms | "
                            f"{processor.label} | {metronome.label}"
                        )
                else:
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
                        metronome.label,
                    )
                    # O Qt mantem a ultima imagem; so frames novos sao enviados.
                    for window, image in zip(windows, (debug, body)):
                        cv2.imshow(window, image)
                last_processed_timestamp = frame_start
            key = cv2.waitKey(1) & 0xFF
            if is_new_frame:
                last_latency_ms = (time.perf_counter() - frame_start) * 1000.0
            if key == ord("q"):
                break
            # Metronomo: teclas globais, valem em todos os modos.
            if key == METRONOME_TOGGLE_KEY:
                metronome.toggle()
                print(metronome.label)
                continue
            if key in METRONOME_TEMPO_KEYS:
                metronome.adjust(METRONOME_TEMPO_KEYS[key])
                print(metronome.label)
                continue
            if key == ord("m"):
                worker.close()
                worker = None
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
                        worker = start_worker()
                        continue
                    previous.close()
                    capturer = replacement

                if capturer is not None:
                    previous_mode = active_mode
                    active_mode = resolve_active_mode(args.mode, capturer)
                    if active_mode != previous_mode:
                        print_key_help(active_mode)
                        if processor is not None:
                            processor.close()
                        processor = make_vision_processor(
                            active_mode,
                            nogpu=args.nogpu,
                            max_distance=args.max_distance,
                            max_connections=args.max_connections,
                            point_deadband=args.point_deadband,
                            show_points=args.show_points,
                            int8=args.int8,
                            edge_style=args.edge_style,
                            grass_density=args.grass_density,
                            grass_height=args.grass_height,
                            grass_gain=args.grass_gain,
                            meadow_density=args.density,
                            meadow_gain=args.gain,
                            landscape_image=args.image,
                            landscape_anchor=args.image_anchor,
                            landscape_regen_delay=args.regen_delay,
                            landscape_regen_time=args.regen_time,
                            landscape_bristle_width=args.bristle_width,
                            landscape_palette=args.palette,
                            plaid_density=args.density,
                            plaid_gain=args.gain,
                            plaid_sensitivity=args.sensitivity,
                            plaid_rotation=args.rotation,
                            plaid_hue=args.hue,
                            pulse_rate=args.pulses,
                            pulse_source=args.pulse_source,
                            silhouette=args.silhouette,
                            plaid_layout=args.layout,
                            render=render,
                        )
                    elif processor is not None:
                        processor.reset()
                    recent_frame_times.clear()
                    last_latency_ms = 0.0
                    last_processed_timestamp = None
                    worker = start_worker()
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
                    calibrate_kinect_background(capturer)
            elif active_mode == "xadrez":
                if key in PLAID_KEY_ADJUSTMENTS:
                    processor.adjust_plaid(**PLAID_KEY_ADJUSTMENTS[key])
                    print(processor.label)
                elif key == SILHOUETTE_TOGGLE_KEY:
                    processor.toggle_silhouette()
                    print(processor.label)
                elif key == LAYOUT_TOGGLE_KEY:
                    processor.toggle_layout()
                    print(processor.label)
                elif key == ord("b") and isinstance(capturer, KinectV1Capturer):
                    # A silhueta usa a mascara de profundidade, que melhora com o fundo.
                    calibrate_kinect_background(capturer)
            elif active_mode == "paisagens" and key in LANDSCAPE_KEY_ADJUSTMENTS:
                processor.adjust_landscape(LANDSCAPE_KEY_ADJUSTMENTS[key])
                print(processor.label)
            elif active_mode == "paisagens" and key in LANDSCAPE_IMAGE_KEYS:
                processor.switch_landscape(LANDSCAPE_IMAGE_KEYS[key])
                print(processor.label)
            elif active_mode == "paisagens" and key == LANDSCAPE_PALETTE_KEY:
                processor.switch_palette()
                print(processor.label)
            elif active_mode == "paisagens" and key in LANDSCAPE_BRISTLE_KEYS:
                processor.adjust_bristles(LANDSCAPE_BRISTLE_KEYS[key])
                print(processor.label)
            elif active_mode == "paisagens" and key in LANDSCAPE_REGEN_KEYS:
                processor.adjust_regeneration(**LANDSCAPE_REGEN_KEYS[key])
                print(processor.label)
            elif active_mode == "paisagens" and key == SILHOUETTE_TOGGLE_KEY:
                processor.toggle_silhouette()
                print(processor.label)
            elif active_mode == "paisagens" and key == ord("b") and isinstance(
                capturer, KinectV1Capturer
            ):
                calibrate_kinect_background(capturer)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario; encerrando captura...", file=sys.stderr)
    finally:
        metronome.close()
        if processor is not None:
            processor.close()
        if worker is not None:
            worker.close()
        if capturer is not None:
            capturer.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
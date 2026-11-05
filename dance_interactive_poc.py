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
Algoritmo: --mode kinect|mog2|optical_flow|conexoes|capim|mato|paisagens|xadrez|caipira|viola|
roda-de-viola|jeca|
jecatatu.
Sem --mode, o modo e caipira em todas as fontes.
Faixa inicial: --min-depth/--max-depth (mm). Atalhos Kinect: a/z diminuem/aumentam
Min Depth; s/x diminuem/aumentam Max Depth, em passos de 100 mm. Pressione b
com a sala vazia para capturar o fundo e recortar apenas o que se move a frente.
O modo Kinect usa profundidade em milimetros (DEPTH_MM).
Atalhos dos modos xadrez e caipira: a/z diminuem/aumentam a densidade, s/x o ganho, d/c a
rotacao do padrao, f/v giram o matiz no disco de cores (anti-horario/horario) e g
liga/desliga a silhueta dos corpos na cor base (com Kinect, b calibra o fundo);
l alterna a malha entre regular e mix (malhas em alturas diferentes). No caipira,
i passa a proxima pintura de fundo.
"""

from __future__ import annotations

import argparse
import colorsys
import functools
import math
import os
import re
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
# O modo caipira comeca como uma copia do xadrez: mesmo codigo, teclas e opcoes.
PLAID_MODES = ("xadrez", "caipira")
# Modo usado quando --mode nao e informado, em qualquer fonte e no --pi.
DEFAULT_MODE = "caipira"
# Sem --hue/--rotation, o caipira parte de laranja e do padrao girado a 45 graus.
CAIPIRA_HUE = 30.0
CAIPIRA_ROTATION = 45.0
# Pessoas na cena, no caipira: regioes separadas da mascara da silhueta (profundidade
# no Kinect, MOG2 nas outras fontes) com area minima, contadas ate o limite. A
# mediana dos ultimos quadros evita que a contagem pisque.
CAIPIRA_MAX_PEOPLE = 5
CAIPIRA_MIN_PERSON_AREA = 0.01  # fracao da area do quadro
CAIPIRA_PEOPLE_WINDOW = 9  # quadros
# Fora do --pi o caipira tambem conta pelo YOLO (cada pessoa detectada), para
# comparar com a mascara; as caixas de cada metodo aparecem na janela de debug.
CAIPIRA_MASK_COLOR = (255, 255, 0)  # BGR: ciano
CAIPIRA_YOLO_COLOR = (255, 0, 255)  # BGR: magenta
# Pulsos do caipira guiados pela desordem das linhas: o deslocamento medio dos
# nos em relacao ao repouso, dividido por CAIPIRA_DISORDER_FULL_PX (0 = malha em
# repouso, 1 = desordem plena). A taxa de nascimento e proporcional a desordem
# (--pulses vezes CAIPIRA_PULSE_RATE_GAIN na desordem plena; malha parada, sem
# pulsos novos) e a velocidade vai de CAIPIRA_PULSE_SPEED[0] a [1] vezes a
# sorteada, valendo tambem para os pulsos que ja estao na linha.
# Calibrado com a figura sintetica do pi_benchmark.py em 640x480: uma andando
# desloca os nos ~10-14 px em media (desordem ~40%), tres ~25 px (~85%).
CAIPIRA_DISORDER_FULL_PX = 30.0
CAIPIRA_PULSE_RATE_GAIN = 3.0
CAIPIRA_PULSE_SPEED = (0.25, 3.0)
# Toada do caipira (--viola): cada pessoa detectada (ate 5) recebe um canal sorteado
# entre os livres; quem some libera o canal. Pessoas pelo YOLO quando carregado,
# senao pela mascara da silhueta.
# Sons: em CAIPIRA_CHANNEL_DIR, como os arpejos (VIOLA_ARPEGGIO_DIR): cada arquivo
# cujo nome contem "canal1" a "canal5". Todos tocam em loop desde a partida,
# sincronizados, mudos ate alguem excitar o canal; por isso precisam ter o mesmo
# andamento e duracao.
CAIPIRA_CHANNEL_DIR = os.path.join(SCRIPT_DIR, "sons", "toada-caipira")
VIOLA_EXTENSIONS = (".mp3", ".wav", ".flac")
VIOLA_SAMPLE_RATE = 44100
# Latencia de saida em segundos: menor responde mais rapido ao gesto, mas o
# callback em Python pode falhar (estalos) com as threads de video ocupadas.
VIOLA_LATENCY = 0.05
# Excitacao das cordas por uma pessoa: deslocamento medio ao repouso, em px, dos nos
# do xadrez dentro da sua caixa, dividido por CAIPIRA_EXCITE_FULL_PX (0 a 1). So
# conta como interacao a partir de CAIPIRA_EXCITE_MIN: estar parada nao muda nada.
# Volume do canal (0 a 1): com interacao, sobe CAIPIRA_VOLUME_RISE por segundo na
# excitacao plena (proporcional a ela), ate 1. Sem interacao por mais de
# CAIPIRA_VOLUME_HOLD s, decai exponencialmente, perdendo metade a cada
# CAIPIRA_VOLUME_HALF_LIFE s: rapido quando alto, devagar perto do silencio (um
# fade linear em dB); abaixo de CAIPIRA_VOLUME_FLOOR, silencia. O canal livre
# continua decaindo do mesmo jeito, e quem o pega comeca do volume em que ele esta.
CAIPIRA_EXCITE_FULL_PX = 20.0
CAIPIRA_EXCITE_MIN = 0.15
CAIPIRA_VOLUME_RISE = 0.25
CAIPIRA_VOLUME_HOLD = 0.5
CAIPIRA_VOLUME_HALF_LIFE = 2.0
CAIPIRA_VOLUME_FLOOR = 0.02
# Rastreio: cada caixa continua a pessoa do quadro anterior com o centro mais
# proximo, ate CAIPIRA_TRACK_MAX_JUMP da diagonal do quadro; quem some por mais de
# CAIPIRA_TRACK_GRACE s libera o canal para a proxima pessoa.
CAIPIRA_TRACK_MAX_JUMP = 0.2
CAIPIRA_TRACK_GRACE = 1.0
CAIPIRA_PLAYER_COLOR = (0, 255, 255)  # BGR: amarelo
# Fundo do caipira: uma das pinturas de --image (padrao: a pasta paisagens/),
# sorteada na partida e ajustada ao quadro como no modo paisagens, escurecida
# para CAIPIRA_BACKGROUND_BRIGHTNESS: as linhas grossas sao compostas com max e
# so aparecem onde sao mais claras que o fundo. A tecla i passa a proxima.
CAIPIRA_BACKGROUND_BRIGHTNESS = 0.5
CAIPIRA_IMAGE_KEY = ord("i")
# Modo viola: pinturas de --image em sequencia embaralhada, cada uma por
# VIOLA_SLIDE_SECONDS, passando a seguinte num fade de VIOLA_FADE_SECONDS (tecla i
# adianta). Cada pessoa detectada (ate 5; YOLO quando carregado, senao a mascara)
# liga um canal de arpejo, o menor livre; o movimento dela aumenta o volume.
# Arpejos: em VIOLA_ARPEGGIO_DIR, cada arquivo cujo nome contem "canal1" a
# "canal5" (mp3, wav ou flac). Todos tocam em loop desde a partida, sincronizados
# por um relogio unico, mudos ate alguem ligar o canal; por isso precisam ter o
# mesmo tom, andamento e duracao. MP3 tem silencio extra no inicio e no fim: com o
# cabecalho LAME/Info (Audacity e ffmpeg o gravam) ele e cortado exatamente; sem
# ele, prefira WAV ou FLAC, ou o loop solua.
VIOLA_ARPEGGIO_DIR = os.path.join(SCRIPT_DIR, "sons", "viola-arpejos")
VIOLA_CHANNELS = 5
VIOLA_SLIDE_SECONDS = 10.0
VIOLA_FADE_SECONDS = 2.0
# Pico de cada canal; a soma passa por um tanh, que so comprime quando varios
# canais altos coincidem.
VIOLA_CHANNEL_PEAK = 0.7
# Volume do canal: VIOLA_CHANNEL_VOLUME[0] com a pessoa parada, [1] com movimento
# de VIOLA_FULL_MOTION px por quadro ou mais (o mesmo movimento da viola do
# caipira), suavizado por VIOLA_MOTION_SMOOTHING a cada quadro.
VIOLA_CHANNEL_VOLUME = (0.25, 1.0)
VIOLA_FULL_MOTION = 4.0
VIOLA_MOTION_SMOOTHING = 0.15
# Mescla do xadrez com as pinturas no modo viola (--blend; tecla e alterna), com a
# mesma fisica de linhas do caipira (laranja, 45 graus, pulsos pela desordem):
# - "tecido": a pintura e deformada junto com as linhas, como um tecido cortado
#   por elas. O deslocamento dos nos vira um campo em grade de VIOLA_WARP_CELL px,
#   espalhado por um blur de VIOLA_WARP_SIGMA celulas (convolucao normalizada:
#   areas entre linhas seguem as linhas vizinhas), ampliado e aplicado com um so
#   cv2.remap. Com a malha em repouso o remap e pulado. A pintura fica com
#   VIOLA_TEXTILE_BRIGHTNESS para as linhas grossas (compostas com max) aparecerem.
# - "janela": a pintura so aparece pelas linhas, sobre ela mesma escurecida para
#   VIOLA_WINDOW_BACKGROUND.
# - "quadros": como o tecido, mas cada quadrado do xadrez segue a deformacao com
#   o seu proprio atraso (taxa sorteada em VIOLA_MOSAIC_LAG por quadro, de quase
#   imediato a ~1 s) e, na troca de pintura, passa a seguinte no seu momento
#   (atraso sorteado ate VIOLA_MOSAIC_STAGGER do fade), como um mosaico.
# - "cores": as linhas tem as cores da pintura, levadas ao brilho maximo
#   (VIOLA_VIVID_MIX) e mais saturadas (VIOLA_VIVID_SATURATION), sobre a pintura a
#   VIOLA_COLOR_BACKGROUND; a versao viva e calculada uma vez por pintura.
# A proxima pintura (e a versao viva, nas cores) e preparada numa thread enquanto
# a atual esta na tela, para a troca nao travar um quadro.
# - "recorte": as linhas sao uma mascara sobre a pintura, e dentro dela a pintura
#   se desloca com as linhas. Parada, a malha nao mostra nada (o recorte coincide
#   com o fundo); com movimento, as tiras deslocadas evidenciam os recortes. Sem
#   movimento so a pintura e copiada; a mascara e desenhada em 1 canal, sem cor,
#   pulsos nem anti-aliasing.
# - "nenhuma": so as pinturas, sem calcular o xadrez.
VIOLA_BLENDS = ("tecido", "quadros", "cores", "janela", "recorte", "nenhuma")
VIOLA_BLEND_KEY = ord("e")
VIOLA_WARP_CELL = 16
VIOLA_WARP_SIGMA = 2.5
VIOLA_TEXTILE_BRIGHTNESS = 0.75
VIOLA_WINDOW_BACKGROUND = 0.2
VIOLA_MOSAIC_LAG = (0.03, 0.6)
VIOLA_MOSAIC_STAGGER = 0.6
VIOLA_VIVID_MIX = 0.6
VIOLA_VIVID_SATURATION = 1.8
VIOLA_COLOR_BACKGROUND = 0.06
VIOLA_COLOR_LIGHT_GAIN = 1.6
# Modo roda-de-viola: cada pessoa (ate 5) vira uma silhueta recortada numa paisagem
# propria, sorteada entre as livres e fixa enquanto ela estiver em cena, sobre
# fundo branco; os arpejos tocam como no modo viola. As silhuetas vem de um modelo
# de segmentacao por instancia (YOLO11n-seg: uma mascara por pessoa, mesmo
# encostadas), em ROLE_SEG_IMGSZ px. Cada mascara e suavizada no tempo
# (ROLE_MASK_SMOOTHING por quadro; 1 = sem suavizacao) e composta em ROLE_ALPHA_SCALE
# da resolucao, so dentro da caixa da pessoa. Borda (ROLE_EDGE_STYLES, tecla e):
# - "poligono": o contorno externo e simplificado (cv2.approxPolyDP) em poucos lados
#   retos e preenchido; ROLE_POLYGON_EPSILON (fracao da altura do quadro) e o
#   desvio maximo da forma: maior = menos lados, mais rustico. Pedacos menores que
#   ROLE_POLYGON_MIN_AREA da area sao descartados; buracos (entre braco e corpo)
#   sao preenchidos.
# - "serrilhado": dentro/fora (limiar 0,5) ampliado sem interpolacao, com os
#   degraus da mascara do modelo (~4 px em 720p).
# - "suave": borda desfocada (ROLE_EDGE_SIGMA px) misturada ao branco.
YOLO_SEG_MODEL = os.path.join(SCRIPT_DIR, "yolo11n-seg.pt")
ROLE_MODE = "roda-de-viola"
ROLE_SEG_IMGSZ = 320
ROLE_SEG_CONFIDENCE = 0.35
ROLE_ALPHA_SCALE = 0.5
ROLE_MASK_SMOOTHING = 0.6
ROLE_EDGE_STYLES = ("poligono", "serrilhado", "suave")
ROLE_EDGE_KEY = ord("e")
ROLE_EDGE_SIGMA = 2.0
ROLE_POLYGON_EPSILON = 0.02
ROLE_POLYGON_MIN_AREA = 0.002
# Estabilidade do poligono: as facetas atuais ficam (so deslocadas com o corpo)
# enquanto a sobreposicao (IoU) com a forma nova for de ROLE_POLYGON_HOLD_IOU ou mais;
# abaixo disso, a pose mudou e o poligono e refeito. A posicao segue o centro da
# mascara com zona morta de ROLE_POLYGON_DEADBAND px da mascara pequena (~3 px em
# 720p): o ruido nao a move, e acima disso ela acompanha o corpo continuamente.
ROLE_POLYGON_HOLD_IOU = 0.92
ROLE_POLYGON_DEADBAND = 1.5
# Cordas da roda: cada pessoa em cena e ligada a cada uma das outras por
# ROLE_STRING_COUNT cordas, como as 10 da viola. As pontas de cada corda sao
# sorteadas dentro de cada silhueta (cabeca, tronco, pernas...), uma vez por par de
# pessoas, e guardadas relativas a caixa de cada uma: acompanham o corpo sem tremer.
# Com 5 pessoas sao 10 ligacoes, 100 cordas. As cordas sao fios de 1 px desenhados
# como tinta semitransparente (ROLE_STRING_OPACITY), que escurece o que esta por
# baixo, para parecerem mais finas que 1 px; a composicao e feita so na regiao das
# cordas. Elas vibram de leve em onda estacionaria (metade delas no modo
# fundamental, metade uma oitava acima, com dois ventres, como os pares oitavados
# do cebolao), cada uma com fase propria. Quanto mais perto os corpos, maior a
# vibracao: a amplitude vai de ROLE_STRING_IDLE (a ROLE_STRING_FAR da diagonal ou
# mais) ao maximo, ROLE_STRING_AMPLITUDE da altura do quadro (a ROLE_STRING_NEAR ou
# menos); e, como numa corda mais curta, a frequencia sobe de ROLE_STRING_HZ[0] a
# [1]. Quem nao e visto ha mais de ROLE_STRING_HOLD s perde as cordas.
ROLE_STRING_COUNT = 10
ROLE_STRING_NEAR = 0.15
ROLE_STRING_FAR = 0.7
ROLE_STRING_AMPLITUDE = 0.012
ROLE_STRING_IDLE = 0.05
ROLE_STRING_HZ = (3.0, 9.0)
ROLE_STRING_SAMPLES = 48
ROLE_STRING_HOLD = 0.2
ROLE_STRING_OPACITY = 0.45
# Modo jeca: as pinturas de --image em sequencia ao fundo (como no modo viola) e,
# por cima, as silhuetas das pessoas (as do roda-de-viola, ate 5) recortadas num
# xadrez proprio de cada uma, com os vaos transparentes: so as faixas sao
# desenhadas e a pintura aparece entre elas. Cada xadrez tem uma paleta de
# JECA_PLAID_COLORS cores (2 a 6) em torno de um matiz por pessoa (JECA_PLAID_HUES,
# pela ordem de chegada): analogas, a complementar e acentos, com luminosidades
# diferentes, em rodizio pelas faixas. O sett e de linhas delicadas (fios de 1 a
# 3 px em 720p, a maioria em grupos de 2 a 4 fios de 1 px), igual nas duas
# direcoes; onde as linhas se cruzam, as cores se misturam. Cada xadrez tem uma
# rotacao sorteada (0 a 180 graus) e fixa. O tecido (periodo de JECA_PLAID_PERIOD
# da altura do quadro) e desenhado uma vez por pessoa e acompanha o corpo como
# uma roupa, com zona morta de JECA_PLAID_DEADBAND px.
JECA_MODE = "jeca"
JECA_PLAID_HUES = (30.0, 200.0, 330.0, 110.0, 265.0)
JECA_PLAID_PERIOD = 0.22
JECA_PLAID_DEADBAND = 3.0
JECA_PLAID_COLORS = (2, 6)
# Modo jecatatu: sobre fundo preto (JECATATU_BACKGROUND), as silhuetas do
# roda-de-viola (poligonais e estabilizadas, ate 5), cada uma recortada numa
# paisagem propria e fixa; por cima de todas, o xadrez do caipira (laranja, 90
# graus, layout regular), sem pulsos de luz. O xadrez reage aos pontos de
# contato de cada pessoa: cabeca, maos e pes, do YOLO de pose. Cada ponto empurra
# as linhas com a sua velocidade, num raio de JECATATU_PUSH_RADIUS da altura do
# quadro, mas so gestos bruscos: abaixo de JECATATU_PUSH_MIN px/quadro (em 720p) o
# ponto nao mexe nas linhas. Um gesto mais brusco ainda (JECATATU_CONTACT_SPEED
# px/quadro ou mais) e um contato: a silhueta da pessoa brilha, um halo que sai do
# contorno (a mascara desfocada em JECATATU_GLOW_RADIUS da altura do quadro, somada por
# cima de tudo num tom claro do matiz do xadrez; dentro do corpo so
# JECATATU_GLOW_INSIDE dele), e o brilho decai com constante de JECATATU_GLOW_SECONDS s; um
# novo contato o reacende. O ponto rearma abaixo de JECATATU_CONTACT_RELEASE e espera
# JECATATU_CONTACT_COOLDOWN s entre contatos.
# Som: cada pessoa em cena mantem tocando o seu canal de arpejo (os do modo viola,
# canal1 a canal5, pela ordem de chegada), em loop e volume constante
# (JECATATU_ARPEGGIO_VOLUME), cada um no seu proprio tempo: um deslocamento sorteado
# no loop, entao pessoas separadas nao casam no compasso. Silhuetas fundidas passam a
# compartilhar o tempo da primeira pessoa do grupo (menor canal) e ficam sincronizadas;
# ao se separar, cada uma volta ao seu. A troca de tempo nao e um salto: o canal faz
# um crossfade de VIOLA_SYNC_FADE s entre a posicao atual e a nova, sem mudar a altura.
JECATATU_MODE = "jecatatu"
JECATATU_BACKGROUND = (0, 0, 0)
JECATATU_ARPEGGIO_VOLUME = 0.8
# Angulo inicial do xadrez no jecatatu (sem --rotation; teclas d/c giram).
JECATATU_ROTATION = 90.0
JECATATU_PUSH_RADIUS = 0.06
JECATATU_PUSH_MIN = 15.0
JECATATU_PUSH_GAIN = 1.0
JECATATU_CONTACT_SPEED = 24.0
JECATATU_CONTACT_RELEASE = 10.0
JECATATU_CONTACT_COOLDOWN = 0.35
JECATATU_GLOW_SECONDS = 0.35
JECATATU_GLOW_RADIUS = 0.045
JECATATU_GLOW_GAIN = 1.6
# Fracao do brilho dentro da silhueta (o resto e o halo em volta).
JECATATU_GLOW_INSIDE = 0.25
# Mistura do matiz do xadrez com branco na cor do brilho (1 = branco).
JECATATU_GLOW_WHITE = 0.55
# Resolucao do calculo do brilho, em relacao a mascara de meia resolucao.
JECATATU_GLOW_SCALE = 0.25
JECATATU_CONTACT_JUMP = 0.12
# Velocidade de cada ponto: mediana dos ultimos JECATATU_VELOCITY_FRAMES passos (um
# salto de um quadro so do modelo de pose e descartado).
JECATATU_VELOCITY_FRAMES = 5
# Confianca media minima de uma pose (abaixo: fantasma do fundo, descartada).
JECATATU_POSE_CONFIDENCE = 0.45
# Silhuetas proximas se fundem: quando a distancia entre as mascaras de duas pessoas
# cai a JECATATU_MERGE_DISTANCE da altura do quadro ou menos, elas viram um poligono so
# (as mascaras unidas e fechadas, com uma ponte sobre o vao), preenchido por outra
# paisagem, diferente das delas; tres ou mais proximas em cadeia formam um grupo so.
# Para separar, a
# distancia precisa passar de JECATATU_SPLIT_DISTANCE (a folga evita que a forma pisque
# no limite). A paisagem do grupo e fixa enquanto ele durar; ao se separarem, cada uma
# volta a sua.
JECATATU_MERGE_DISTANCE = 0.04
JECATATU_SPLIT_DISTANCE = 0.07
# Pontos COCO: cabeca = media de nariz, olhos e orelhas; maos = pulsos; pes = tornozelos.
JECATATU_KEYPOINTS = (
    ("cabeca", (0, 1, 2, 3, 4)),
    ("mao", (9,)),
    ("mao", (10,)),
    ("pe", (15,)),
    ("pe", (16,)),
)
JECATATU_PART_COLORS = {"cabeca": (0, 255, 255), "mao": (255, 0, 255), "pe": (0, 255, 0)}
# Rampas de volume no audio, em unidades de volume por segundo: entra em ~0,3 s e
# sai em ~1,5 s, sem estalos.
VIOLA_VOLUME_RISE = 1.0 / 0.3
VIOLA_VOLUME_FALL = 1.0 / 1.5
# Troca de tempo de um canal (jecatatu): crossfade, em segundos, entre a posicao atual
# no loop e a nova.
VIOLA_SYNC_FADE = 2.0
# Atraso de decodificacao do MP3 (minimp3), somado ao do codificador no corte.
MP3_DECODER_DELAY = 529
# Comprimentos em fracao da altura do quadro, divididos por --density.
DEFAULT_PLAID_DENSITY = 1.0
DEFAULT_PLAID_GAIN = 1.0
DEFAULT_PLAID_ROTATION = 0.0
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
}
# Acima de ~4x o numero de linhas (e o custo do desenho) passa de ~4x o padrao.
PLAID_DENSITY_RANGE = (0.25, 4.0)
# No mix, densidade maxima: acima disso a camada distante vira textura e o custo
# do desenho dispara (ela ja tem o dobro da densidade global).
PLAID_MIX_MAX_DENSITY = 2.0
PLAID_MAX_GAIN = 5.0
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
PI_MODES = ("caipira", "xadrez", "paisagens")
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

    def displacement_total(self) -> tuple[float, int]:
        """Soma do deslocamento dos nos internos ao repouso (px) e quantos sao."""
        inner = self._displacement[:, 1:-1]
        total = float(np.hypot(inner[..., 0], inner[..., 1]).sum())
        return total, inner.shape[0] * inner.shape[1]

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
        # Multiplica a velocidade de todos os pulsos (o caipira a liga a desordem).
        self.speed_scale = 1.0
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
        self._family = np.concatenate((self._family, family))
        self._line = np.concatenate((self._line, line))
        self._position = np.concatenate((self._position, np.zeros(count, np.float32)))
        self._speed = np.concatenate((self._speed, speed.astype(np.float32)))
        self._tail = np.concatenate((self._tail, tail.astype(np.float32)))
        self._reverse = np.concatenate((self._reverse, self._rng.random(count) < 0.5))

    def advance(self, dt: float) -> None:
        """Move os pulsos dt segundos, descarta os que sairam e sorteia novos."""
        self._position += self._speed * (self.speed_scale * dt)
        alive = self._position - self._tail < self._sample_counts[self._family] - 1
        for name in ("_family", "_line", "_position", "_speed", "_tail", "_reverse"):
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
        back = head + np.where(self._reverse, self._tail, -self._tail)
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

    def displacement_total(self) -> tuple[float, int]:
        totals = [family.displacement_total() for family in self._families]
        return sum(total for total, _ in totals), sum(count for _, count in totals)

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
        # Caipira: taxa e velocidade dos pulsos seguem a desordem (0 a 1).
        self.disorder_pulses = False
        self.disorder = 0.0
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
        self._pulse_shares = [
            pulse_rate * len(field) / total if total else 0.0 for field, _ in self._layers
        ]
        for (field, _), share in zip(self._layers, self._pulse_shares):
            field.pulses.rate = share
        self._scratch = np.zeros((height, width, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return sum(len(field) for field, _ in self._layers)

    @property
    def pulse_count(self) -> int:
        return sum(len(field.pulses) for field, _ in self._layers)

    def node_motion(self) -> tuple[np.ndarray, np.ndarray]:
        """Posicao atual e deslocamento de todos os nos, (nos, 2) cada."""
        positions, moves = [], []
        for field, _ in self._layers:
            for family in field._families:
                if len(family):
                    moves.append(family._displacement.reshape(-1, 2))
                    positions.append(family._rest.reshape(-1, 2) + moves[-1])
        if not moves:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty
        return np.concatenate(positions), np.concatenate(moves)

    def reset(self) -> None:
        for field, _ in self._layers:
            field.reset()

    def update(
        self, flow: Optional[np.ndarray], scale: np.ndarray, threshold: float
    ) -> None:
        for field, push in self._layers:
            field.update(flow, scale, threshold, self.gain * push)

    def advance(self, dt: float) -> None:
        if self.disorder_pulses:
            self._drive_pulses()
        for field, _ in self._layers:
            field.pulses.advance(dt)

    def _drive_pulses(self) -> None:
        """Taxa e velocidade dos pulsos proporcionais a desordem das linhas."""
        totals = [field.displacement_total() for field, _ in self._layers]
        count = sum(nodes for _, nodes in totals)
        mean = sum(total for total, _ in totals) / count if count else 0.0
        self.disorder = min(1.0, mean / CAIPIRA_DISORDER_FULL_PX)
        slow, fast = CAIPIRA_PULSE_SPEED
        for (field, _), share in zip(self._layers, self._pulse_shares):
            field.pulses.rate = share * CAIPIRA_PULSE_RATE_GAIN * self.disorder
            field.pulses.speed_scale = slow + (fast - slow) * self.disorder

    def draw_mask(self, mask: np.ndarray) -> None:
        """Linhas em 255 numa mascara de um canal: sem cor, pulsos nem anti-aliasing,
        bem mais barato que draw()."""
        for field, _ in self._layers:
            _, batches = field.line_batches(self.hue)
            for thickness, _, lines in batches:
                cv2.polylines(mask, lines, False, 255, thickness, cv2.LINE_8)

    def draw(self, image: np.ndarray) -> None:
        drawn = [field.line_batches(self.hue) for field, _ in self._layers]
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
        self, width: int, height: int, image: np.ndarray, anchor: float, index: int = 0
    ) -> None:
        self.size = (width, height)
        self.gain = 1.0
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
        noise = np.random.default_rng().standard_normal((height, width)).astype(np.float32)
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


def make_jeca_plaid(
    rng: np.random.Generator, hue: float, height: int
) -> tuple[int, list[tuple[int, int, int]], np.ndarray]:
    """Sett do xadrez do modo jeca: (periodo em px, faixas (inicio, largura, cor), paleta BGR).

    Paleta de JECA_PLAID_COLORS cores em torno do matiz dado; linhas delicadas (fios
    soltos e grupos de fios de 1 px) separadas por vaos transparentes, com as cores
    em rodizio para todas aparecerem.
    """
    period = max(24, round(JECA_PLAID_PERIOD * height))
    unit = height / 720.0
    count = int(rng.integers(JECA_PLAID_COLORS[0], JECA_PLAID_COLORS[1] + 1))
    # Matiz da pessoa, analogas, a complementar e acentos, em luminosidades variadas.
    offsets = (0.0, 30.0, -30.0, 180.0, 65.0, -115.0)[:count]
    palette = []
    for offset in offsets:
        red, green, blue = colorsys.hls_to_rgb(
            ((hue + offset + rng.uniform(-8.0, 8.0)) % 360.0) / 360.0,
            rng.uniform(0.32, 0.8),
            rng.uniform(0.55, 0.95),
        )
        palette.append((blue * 255.0, green * 255.0, red * 255.0))
    thin = max(1, round(unit))
    stripes = []
    order = list(rng.permutation(count))
    cursor = 0
    while cursor < period:
        # Cores em rodizio embaralhado: todas aparecem no sett.
        if not order:
            order = list(rng.permutation(count))
        color = int(order.pop())
        if rng.random() < 0.3:
            widths = [max(1, round(float(rng.choice((2, 3))) * unit))]
        else:
            widths = [thin] * int(rng.integers(2, 5))
        for width in widths:
            if cursor + width > period:
                break
            stripes.append((cursor, width, color))
            cursor += width + max(1, round(2 * unit))
        cursor += max(2, round(float(rng.uniform(6.0, 18.0)) * unit))
    return period, stripes, np.array(palette, dtype=np.float32)


def jeca_plaid_lattice(period: int, angle: float) -> tuple[np.ndarray, float]:
    """Vetor inteiro (a, b) mais proximo de period no angulo dado, e o angulo dele.

    Com um passo de repeticao inteiro, o tecido girado se repete exatamente em
    pixels inteiros (a, b) e (-b, a), e o recorte nunca salta.
    """
    radians = math.radians(angle)
    lattice = np.rint((period * math.cos(radians), period * math.sin(radians))).astype(int)
    if not lattice.any():
        lattice = np.array((period, 0))
    return lattice, math.degrees(math.atan2(lattice[1], lattice[0]))


def render_jeca_plaid(
    period: int,
    stripes: list[tuple[int, int, int]],
    palette: np.ndarray,
    lattice: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tecido de width x height com o sett girado ao longo do vetor inteiro lattice:
    cores BGR e linhas (booleano).

    As linhas sao desenhadas ja inclinadas (cv2.line, LINE_8), e nao giradas depois
    como imagem: fios de 1 px continuam continuos e nitidos em qualquer angulo. Cada
    repeticao e o mesmo traco deslocado por um vetor inteiro, entao o tecido se
    repete exatamente. Nos cruzamentos as duas cores se misturam.
    """
    step = float(np.hypot(*lattice))
    first = lattice.astype(np.float64) / step
    second = np.array((-first[1], first[0]))
    # Posicoes do sett proporcionais ao passo real (que difere pouco de period).
    ratio = step / period
    center = np.array((width // 2, height // 2), dtype=np.float64)
    reach = math.hypot(width, height) / 2.0 + step
    repeats = range(-math.ceil(reach / step) - 1, math.ceil(reach / step) + 1)
    shift = 4
    families = []
    lattices = (lattice, np.array((-lattice[1], lattice[0])))
    for (across, along), vector in zip(((first, second), (second, first)), lattices):
        # Indice da cor + 1 de cada pixel (0 = sem linha).
        layer = np.zeros((height, width), dtype=np.uint8)
        for repeat in repeats:
            for start, line_width, color in stripes:
                # Parte inteira (repeticoes do vetor) separada: tracos identicos.
                middle = (
                    center + repeat * vector + across * ((start + line_width / 2.0) * ratio)
                )
                ends = [
                    tuple(int(round(value * (1 << shift))) for value in point)
                    for point in (middle - along * reach, middle + along * reach)
                ]
                cv2.line(layer, ends[0], ends[1], color + 1, line_width, cv2.LINE_8, shift)
        families.append(layer)
    # Cor de cada combinacao (linha de uma familia, da outra, das duas) numa tabela:
    # indice = linha * (cores + 1) + coluna, aplicada com cv2.LUT.
    count = len(palette) + 1
    combined = cv2.add(cv2.multiply(families[0], count), families[1])
    table = np.zeros((256, 3), dtype=np.float32)
    for row in range(count):
        for column in range(count):
            if row and column:
                color = (palette[row - 1] + palette[column - 1]) / 2.0
            elif row or column:
                color = palette[max(row, column) - 1]
            else:
                continue
            table[row * count + column] = color
    texture = cv2.LUT(cv2.merge((combined, combined, combined)), table.astype(np.uint8)[None])
    return texture, combined > 0


class PersonSegmenter:
    """Uma mascara por pessoa (YOLO-seg do Ultralytics), na GPU NVIDIA quando ha."""

    def __init__(self, nogpu: bool = False) -> None:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                f"O modo {ROLE_MODE} requer ultralytics/PyTorch; instale com "
                "python -m pip install -r requirements.txt"
            ) from exc
        if not os.path.isfile(YOLO_SEG_MODEL):
            print(
                f"Primeiro uso do modo {ROLE_MODE}: baixando "
                f"{os.path.basename(YOLO_SEG_MODEL)} (~6 MB) para {SCRIPT_DIR}; precisa de "
                "internet.",
                flush=True,
            )
        cuda = bool(torch.cuda.is_available()) and not nogpu
        self._device = 0 if cuda else "cpu"
        self.label = "YOLO11n-seg / " + ("NVIDIA GPU" if cuda else "CPU")
        self._model = YOLO(YOLO_SEG_MODEL)

    def people(
        self, frame_bgr: np.ndarray
    ) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
        """Caixas (x0, y0, x1, y1) no quadro e mascaras (0 ou 1) do quadro inteiro,
        na escala do modelo, sem as bordas do letterbox."""
        result = self._model(
            frame_bgr,
            imgsz=ROLE_SEG_IMGSZ,
            classes=[0],
            conf=ROLE_SEG_CONFIDENCE,
            max_det=YOLO_MAX_DETECTIONS,
            device=self._device,
            verbose=False,
        )[0]
        if result.masks is None:
            return [], []
        masks = result.masks.data.cpu().numpy().astype(np.float32)
        height, width = frame_bgr.shape[:2]
        mask_height, mask_width = masks.shape[1:]
        ratio = min(mask_height / height, mask_width / width)
        inner_width, inner_height = round(width * ratio), round(height * ratio)
        # Mesmo arredondamento do letterbox do Ultralytics.
        left = round((mask_width - inner_width) / 2 - 0.1)
        top = round((mask_height - inner_height) / 2 - 0.1)
        masks = masks[:, top : top + inner_height, left : left + inner_width]
        boxes = [
            tuple(int(round(value)) for value in box)
            for box in result.boxes.xyxy.cpu().numpy().tolist()
        ]
        return boxes, list(masks)


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
        "caipira": "Caipira (Optical Flow)",
        "viola": "Viola (arpejos)",
        ROLE_MODE: "Roda de viola (silhuetas)",
        JECA_MODE: "Jeca (silhuetas em xadrez)",
        JECATATU_MODE: "Jecatatu (silhuetas, xadrez e arpejos)",
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
        plaid_rotation: Optional[float] = None,
        plaid_hue: Optional[float] = None,
        pulse_rate: float = DEFAULT_PULSE_RATE,
        silhouette: bool = False,
        plaid_layout: str = "mix",
        render: PlaidRender = PlaidRender(),
        people_yolo: bool = False,
        arpeggios: Optional[ViolaArpeggios] = None,
        viola_blend: str = VIOLA_BLENDS[0],
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
        # Alternada pela tecla g na thread principal; o worker so le o valor.
        self.silhouette = silhouette
        # Criado na primeira vez que a silhueta precisa do MOG2 (sem Kinect).
        self._silhouette_mog2 = None
        # Pessoas detectadas no caipira; o worker escreve, a thread principal le.
        # Desordem das linhas do caipira (0 a 1), que guia os pulsos.
        self.line_disorder = 0.0
        self.people_count = 0
        self._people_history: deque[int] = deque(maxlen=CAIPIRA_PEOPLE_WINDOW)
        # Caixas (x0, y0, x1, y1) do ultimo quadro, para o debug, e origem da
        # mascara contada: "Kinect" (profundidade) ou "MOG2".
        self._people_boxes: list[tuple[int, int, int, int]] = []
        self._people_source = "MOG2"
        # Contagem pelo YOLO; None quando ele nao esta carregado.
        self.people_count_yolo: Optional[int] = None
        self._people_yolo_history: deque[int] = deque(maxlen=CAIPIRA_PEOPLE_WINDOW)
        self._people_yolo_boxes: list[tuple[int, int, int, int]] = []
        # Pessoas rastreadas, cada uma com o seu canal (caipira, viola e roda).
        self._viola_players: list[ViolaPlayer] = []
        # Canais em loop: a toada no caipira, os arpejos na viola, roda e jecatatu.
        self.arpeggios = (
            arpeggios if mode in ("caipira", "viola", ROLE_MODE, JECATATU_MODE) else None
        )
        # Caipira: volume (0 a 1) de cada canal e o instante da ultima interacao,
        # guardados por canal para quem sai deixar o canal decaindo.
        self._toada_levels: dict[int, float] = {}
        self._toada_excited: dict[int, float] = {}
        self._toada_clock: Optional[float] = None
        # Roda de viola: segmentacao por pessoa e fundo branco guardado.
        self._segmenter: Optional[PersonSegmenter] = None
        self._white: Optional[np.ndarray] = None
        # Jecatatu: pontos de contato do quadro anterior e fundo guardado.
        self._contacts: list[dict] = []
        self._black: Optional[np.ndarray] = None
        self.contact_count = 0
        # Pares de silhuetas somadas (slots) e a paisagem de cada grupo.
        self._merged_pairs: set[tuple[int, int]] = set()
        self._group_paintings: dict[frozenset[int], int] = {}
        # Forma fundida de cada grupo (mascara e poligono estabilizado).
        self._group_shapes: dict[frozenset[int], SilhouetteShape] = {}
        # Pontas das cordas de cada par de pessoas (relativas as caixas), por par.
        self._string_pairs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        # Alternado pela tecla e na thread principal; o worker so le o valor.
        self.role_edge = ROLE_EDGE_STYLES[0]
        self.channel_volumes: dict[int, float] = {}
        self._slide_cache: dict[tuple[int, int, int, bool], np.ndarray] = {}
        self._prefetch_thread: Optional[threading.Thread] = None
        self._slide_queue: list[int] = []
        self._slide_start: Optional[float] = None
        self._slide_skip = False
        if viola_blend not in VIOLA_BLENDS:
            raise ValueError(f"Mescla desconhecida: {viola_blend}")
        # Alternada pela tecla e na thread principal; o worker so le o valor.
        self.viola_blend = viola_blend
        # Coordenadas dos centros da grade do tecido, refeitas quando o quadro muda.
        self._warp_size: Optional[tuple[int, int, int, int]] = None
        self._warp_grid: Optional[np.ndarray] = None
        # Quadros: quadrado de cada celula/pixel por malha, atraso de cada quadrado
        # (taxa por quadro e inicio na troca) e o deslocamento atrasado.
        self._mosaic_key: Optional[tuple[int, int, int]] = None
        self._mosaic_maps: tuple[np.ndarray, np.ndarray] = (np.empty(0), np.empty(0))
        self._mosaic_shift: Optional[np.ndarray] = None
        mosaic_rng = np.random.default_rng()
        self._mosaic_rates = mosaic_rng.uniform(*VIOLA_MOSAIC_LAG, 256).astype(np.float32)
        self._mosaic_offsets = mosaic_rng.uniform(0.0, VIOLA_MOSAIC_STAGGER, 256)
        self._viola_now = 0.0
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
        if mode in ("paisagens", "caipira", "viola", ROLE_MODE, JECA_MODE, JECATATU_MODE):
            for path in list_landscape_images(landscape_image):
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Nao foi possivel abrir a imagem {path}")
                name = os.path.splitext(os.path.basename(path))[0]
                self._landscape_images.append((name, image))
        # Ajustaveis por teclado (adjust_plaid) enquanto o worker processa: o
        # worker so le os valores e recria a malha quando densidade/rotacao mudam.
        self.plaid_density = min(plaid_density, self._max_plaid_density())
        if mode in PLAID_MODES and self.plaid_density < plaid_density:
            print(
                f"Aviso: no layout mix a densidade e limitada a {PLAID_MIX_MAX_DENSITY}; "
                f"usando {self.plaid_density} em vez de {plaid_density}.",
                file=sys.stderr,
            )
        self.plaid_gain = plaid_gain
        if plaid_rotation is None:
            plaid_rotation = (
                JECATATU_ROTATION
                if mode == JECATATU_MODE
                else CAIPIRA_ROTATION
                if mode in ("caipira", "viola")
                else DEFAULT_PLAID_ROTATION
            )
        self.plaid_rotation = plaid_rotation % 360.0
        self._rng = np.random.default_rng()
        if mode in ("caipira", "viola", JECA_MODE):
            self.landscape_index = int(self._rng.integers(len(self._landscape_images)))
        if mode in ("viola", JECA_MODE):
            self._slide_next = self._next_slide_index()
        # Fundo do caipira ja ajustado e escurecido, por (indice, largura, altura).
        self._background_key: Optional[tuple[int, int, int]] = None
        self._background: Optional[np.ndarray] = None
        # Sem --hue, o matiz base e sorteado no xadrez e laranja no caipira.
        if plaid_hue is None:
            plaid_hue = (
                CAIPIRA_HUE
                if mode in ("caipira", "viola", JECATATU_MODE)
                else float(self._rng.uniform(0.0, 360.0))
            )
        self.plaid_hue = plaid_hue % 360.0
        # Os padroes do xadrez (sett, fase de cada direcao e desvio de angulo, um
        # por camada do mix) sao sorteados ja aqui, em fracoes da altura do quadro,
        # e mantidos ao mudar densidade, rotacao e disposicao.
        pattern_count = (
            PLAID_PATTERN_COUNT if mode in (*PLAID_MODES, "viola", JECATATU_MODE) else 0
        )
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
        elif mode in (ROLE_MODE, JECA_MODE):
            self._segmenter = PersonSegmenter(nogpu)
        elif mode == JECATATU_MODE:
            self._segmenter = PersonSegmenter(nogpu)
            try:
                self._load_yolo(nogpu, int8)
            except Exception as exc:
                print(
                    f"Aviso: sem YOLO de pose no {mode}, o xadrez nao reage aos contatos: {exc}",
                    file=sys.stderr,
                )
        elif mode in ("caipira", "viola") and people_yolo:
            try:
                self._load_yolo(nogpu, int8)
                self.people_count_yolo = 0
            except Exception as exc:
                print(
                    f"Aviso: contagem de pessoas pelo YOLO desativada no {mode}: {exc}",
                    file=sys.stderr,
                )

    @property
    def label(self) -> str:
        if self.mode == "conexoes" and self._yolo_backend:
            return f"YOLOv8 Pose [{self._yolo_backend}]"
        if self.mode == "viola":
            channels = " ".join(
                f"canal{channel} {volume:.0%}"
                for channel, volume in sorted(self.channel_volumes.items())
                if volume > 0
            )
            return (
                f"Viola: {self._landscape_images[self.landscape_index][0]}"
                f" | pessoas {self._people_source} {self.people_count}"
                + ("" if self.people_count_yolo is None else f" / YOLO {self.people_count_yolo}")
                + f" | {channels or 'canais mudos'}"
                + f" | {self.viola_blend}"
                + (
                    f" | desordem {self.line_disorder:.0%}"
                    if self.viola_blend != "nenhuma"
                    else ""
                )
                + (" | sem arpejos" if self.arpeggios is None else "")
            )
        if self.mode == JECATATU_MODE:
            return (
                f"Jecatatu [{self._segmenter.label}]: pessoas {self.people_count}"
                f" | pontos {len(self._contacts)} | contatos {self.contact_count}"
                f" | unioes {len(self._group_paintings)}"
                f" | borda {self.role_edge}"
                + (" | sem pose" if self._yolo is None else "")
                + f" | {self._arpeggio_label()}"
            )
        if self.mode == JECA_MODE:
            return (
                f"Jeca [{self._segmenter.label}]: "
                f"{self._landscape_images[self.landscape_index][0]}"
                f" | pessoas {self.people_count} | borda {self.role_edge}"
            )
        if self.mode == ROLE_MODE:
            channels = " ".join(
                f"canal{channel} {volume:.0%}"
                for channel, volume in sorted(self.channel_volumes.items())
                if volume > 0
            )
            paintings = " ".join(
                f"canal{player.slot}={self._landscape_images[player.painting][0]}"
                for player in self._viola_players
                if player.painting is not None
            )
            return (
                f"Roda de viola [{self._segmenter.label}]: pessoas {self.people_count}"
                + f" | borda {self.role_edge}"
                + f" | {channels or 'canais mudos'}"
                + (f" | {paintings}" if paintings else "")
                + (" | sem arpejos" if self.arpeggios is None else "")
            )
        if self.mode in PLAID_MODES:
            return (
                f"{self.mode.capitalize()}: dens. {self.plaid_density:.2f} | "
                f"ganho {self.plaid_gain:.1f} | "
                f"rot. {self.plaid_rotation:.0f} | matiz {self.plaid_hue:.0f}"
                + (" | silhueta" if self.silhouette else "")
                + f" | {self.plaid_layout}"
                + (
                    f" | fundo {self._landscape_images[self.landscape_index][0]}"
                    f" ({self.landscape_index + 1}/{len(self._landscape_images)})"
                    if self.mode == "caipira"
                    else ""
                )
                + (
                    f" | desordem {self.line_disorder:.0%}"
                    f" | pessoas {self._people_source} {self.people_count}"
                    + (
                        ""
                        if self.people_count_yolo is None
                        else f" / YOLO {self.people_count_yolo}"
                    )
                    + " | "
                    + (
                        " ".join(
                            f"canal{channel} {volume:.0%}"
                            for channel, volume in sorted(self.channel_volumes.items())
                            if volume > 0
                        )
                        or "canais mudos"
                    )
                    + (" | sem toada" if self.arpeggios is None else "")
                    if self.mode == "caipira"
                    else ""
                )
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
        elif self.mode in (
            "capim", "mato", "paisagens", "viola", ROLE_MODE, JECA_MODE, JECATATU_MODE,
            *PLAID_MODES,
        ):
            # Mantem o capim/xadrez sorteado; um quadro de outro tamanho o recria.
            self._previous_gray = None
            self._field_clock = None
            self._people_history.clear()
            self.people_count = 0
            self._people_yolo_history.clear()
            self._viola_players = []
            # Os canais da toada seguem decaindo; so o passo de tempo recomeca.
            self._toada_clock = None
            self._contacts = []
            if self.people_count_yolo is not None:
                self.people_count_yolo = 0
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
        if self.mode == JECA_MODE:
            return self._process_jeca(frame_bgr, body_mask)
        if self.mode == JECATATU_MODE:
            return self._process_jecatatu(frame_bgr, body_mask)
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

        if self.mode in (
            "optical_flow", "capim", "mato", "paisagens", "viola", ROLE_MODE, *PLAID_MODES
        ):
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
            elif self.mode == "viola":
                result = self._process_viola(frame_bgr, small_gray, body_mask)
            elif self.mode == ROLE_MODE:
                result = self._process_role(frame_bgr, small_gray, body_mask)
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
            return LandscapeCanvas(width, height, image, self._crop_anchor(image), index)
        if self.mode == "mato":
            return MeadowField(width, height, self.meadow_density, self._rng)
        if self.mode in (*PLAID_MODES, "viola", JECATATU_MODE):
            return PlaidStack(
                width,
                height,
                self._plaid_patterns,
                self.plaid_layout,
                self.plaid_density,
                self.plaid_rotation,
                # No jecatatu os pulsos so saem dos contatos.
                0.0 if self.mode == JECATATU_MODE else self.pulse_rate,
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

    def _count_people(self, mask: np.ndarray) -> None:
        """Conta os corpos da mascara da silhueta, ate CAIPIRA_MAX_PEOPLE."""
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = CAIPIRA_MIN_PERSON_AREA * mask.shape[0] * mask.shape[1]
        bodies = stats[1:][stats[1:, cv2.CC_STAT_AREA] >= min_area]
        self._people_history.append(min(len(bodies), CAIPIRA_MAX_PEOPLE))
        self.people_count = sorted(self._people_history)[len(self._people_history) // 2]
        self._people_boxes = [
            (
                round(x / SILHOUETTE_SCALE),
                round(y / SILHOUETTE_SCALE),
                round((x + w) / SILHOUETTE_SCALE),
                round((y + h) / SILHOUETTE_SCALE),
            )
            for x, y, w, h in bodies[:, :4].tolist()
        ]

    def _count_people_yolo(self, frame_bgr: np.ndarray) -> None:
        """Conta as pessoas detectadas pelo YOLO, ate CAIPIRA_MAX_PEOPLE."""
        points, confidence = self._yolo_keypoints(frame_bgr)
        self._people_yolo_history.append(min(len(points), CAIPIRA_MAX_PEOPLE))
        history = sorted(self._people_yolo_history)
        self.people_count_yolo = history[len(history) // 2]
        # Caixa de cada pessoa: extremos dos keypoints confiaveis.
        boxes = []
        for index, person in enumerate(points):
            if confidence is not None:
                person = person[confidence[index] >= YOLO_MIN_KEYPOINT_CONFIDENCE]
            if len(person):
                (x0, y0), (x1, y1) = person.min(axis=0), person.max(axis=0)
                boxes.append((round(x0), round(y0), round(x1), round(y1)))
        self._people_yolo_boxes = boxes

    def _track_players(
        self,
        boxes: list[tuple[int, int, int, int]],
        flow: Optional[np.ndarray],
        scale: np.ndarray,
        size: tuple[int, int],
        slots: list[int],
        random_slot: bool,
    ) -> float:
        """Rastreia as pessoas e da a cada nova um slot livre (par ou canal).

        O slot e sorteado (random_slot) ou o menor livre; quem some por mais de
        CAIPIRA_TRACK_GRACE o libera. Mede o movimento de quem esta no quadro e
        devolve o instante do quadro.
        """
        now = self.clock()
        self._viola_now = now
        boxes = sorted(
            boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True
        )[:CAIPIRA_MAX_PEOPLE]
        players = self._viola_players
        # Associacao gulosa pela menor distancia entre centros.
        max_jump = CAIPIRA_TRACK_MAX_JUMP * math.hypot(*size)
        candidates = sorted(
            (
                math.hypot(
                    player.center[0] - (box[0] + box[2]) / 2.0,
                    player.center[1] - (box[1] + box[3]) / 2.0,
                ),
                player_index,
                box_index,
            )
            for player_index, player in enumerate(players)
            for box_index, box in enumerate(boxes)
        )
        matched_players, matched_boxes = set(), set()
        for distance, player_index, box_index in candidates:
            if distance > max_jump:
                break
            if player_index in matched_players or box_index in matched_boxes:
                continue
            players[player_index].place(boxes[box_index], now)
            matched_players.add(player_index)
            matched_boxes.add(box_index)
        players = [player for player in players if now - player.seen <= CAIPIRA_TRACK_GRACE]
        taken = {player.slot for player in players}
        free = [slot for slot in sorted(slots) if slot not in taken]
        for box_index, box in enumerate(boxes):
            if box_index not in matched_boxes and free:
                index = int(self._rng.integers(len(free))) if random_slot else 0
                players.append(ViolaPlayer(box, free.pop(index), now))
        magnitude = None if flow is None else np.hypot(flow[..., 0], flow[..., 1])
        for player in players:
            player.motion = 0.0
            if magnitude is None or player.seen != now:
                continue
            x0, y0, x1, y1 = player.box
            # Caixa na grade do fluxo; o movimento volta a px do quadro original.
            cells = magnitude[
                max(0, int(y0 * scale[1])) : max(1, int(math.ceil(y1 * scale[1]))),
                max(0, int(x0 * scale[0])) : max(1, int(math.ceil(x1 * scale[0]))),
            ]
            if cells.size:
                player.motion = float(cells.mean()) / float(scale[0])
        self._viola_players = players
        return now

    def _play_toada(
        self,
        boxes: list[tuple[int, int, int, int]],
        flow: Optional[np.ndarray],
        scale: np.ndarray,
        size: tuple[int, int],
        field: PlaidStack,
    ) -> None:
        """Caipira: da um canal sorteado a cada pessoa nova; as cordas que ela excita
        sobem o volume do canal, que decai sem interacao."""
        channels = (
            sorted(self.arpeggios.channels)
            if self.arpeggios is not None
            else list(range(1, VIOLA_CHANNELS + 1))
        )
        now = self._track_players(boxes, flow, scale, size, channels, True)
        dt = 0.0 if self._toada_clock is None else min(now - self._toada_clock, 0.1)
        self._toada_clock = now
        positions, moves = field.node_motion()
        displacement = np.hypot(moves[:, 0], moves[:, 1])
        excitations = {}
        for player in self._viola_players:
            player.excitation = 0.0
            if player.seen != now:
                continue
            x0, y0, x1, y1 = player.box
            inside = (
                (positions[:, 0] >= x0) & (positions[:, 0] <= x1)
                & (positions[:, 1] >= y0) & (positions[:, 1] <= y1)
            )
            if inside.any():
                player.excitation = min(
                    1.0, float(displacement[inside].mean()) / CAIPIRA_EXCITE_FULL_PX
                )
                excitations[player.slot] = player.excitation
        decay = 0.5 ** (dt / CAIPIRA_VOLUME_HALF_LIFE)
        for channel in channels:
            level = self._toada_levels.get(channel, 0.0)
            excitation = excitations.get(channel, 0.0)
            if excitation >= CAIPIRA_EXCITE_MIN:
                level = min(1.0, level + CAIPIRA_VOLUME_RISE * excitation * dt)
                self._toada_excited[channel] = now
            elif now - self._toada_excited.get(channel, -math.inf) > CAIPIRA_VOLUME_HOLD:
                level *= decay
                if level < CAIPIRA_VOLUME_FLOOR:
                    level = 0.0
            self._toada_levels[channel] = level
        self.channel_volumes = {channel: self._toada_levels[channel] for channel in channels}
        if self.arpeggios is not None:
            for channel, volume in self.channel_volumes.items():
                self.arpeggios.set_volume(channel, volume)

    def _mix_arpeggios(
        self,
        boxes: list[tuple[int, int, int, int]],
        flow: Optional[np.ndarray],
        scale: np.ndarray,
        size: tuple[int, int],
    ) -> None:
        """Viola: cada pessoa liga o menor canal livre; o movimento aumenta o volume."""
        channels = (
            sorted(self.arpeggios.channels)
            if self.arpeggios is not None
            else list(range(1, VIOLA_CHANNELS + 1))
        )
        now = self._track_players(boxes, flow, scale, size, channels, False)
        low, high = VIOLA_CHANNEL_VOLUME
        volumes = dict.fromkeys(channels, 0.0)
        for player in self._viola_players:
            # Quem sumiu por pouco tempo mantem o nivel ate o rastreio desistir.
            if player.seen == now:
                level = min(1.0, player.motion / VIOLA_FULL_MOTION)
                player.level += VIOLA_MOTION_SMOOTHING * (level - player.level)
            volumes[player.slot] = low + (high - low) * player.level
        self.channel_volumes = volumes
        if self.arpeggios is not None:
            for channel, volume in volumes.items():
                self.arpeggios.set_volume(channel, volume)

    def _slide(self, index: int, width: int, height: int, vivid: bool = False) -> np.ndarray:
        """Pintura ajustada ao quadro (ou a versao viva), guardada por chave."""
        key = (index, width, height, vivid)
        if key not in self._slide_cache:
            if vivid:
                # Em HSV: saturacao multiplicada e brilho puxado para o maximo.
                hue, saturation, value = cv2.split(
                    cv2.cvtColor(self._slide(index, width, height), cv2.COLOR_BGR2HSV)
                )
                saturation = cv2.convertScaleAbs(saturation, alpha=VIOLA_VIVID_SATURATION)
                value = cv2.convertScaleAbs(
                    value, alpha=1.0 - VIOLA_VIVID_MIX, beta=255.0 * VIOLA_VIVID_MIX
                )
                self._slide_cache[key] = cv2.cvtColor(
                    cv2.merge((hue, saturation, value)), cv2.COLOR_HSV2BGR
                )
            else:
                image = self._landscape_images[index][1]
                self._slide_cache[key] = fit_cover(
                    image, width, height, self._crop_anchor(image)
                )
        return self._slide_cache[key]

    def _prefetch_slide(self, index: int, width: int, height: int, vivid: bool) -> None:
        """Prepara a proxima pintura numa thread, para a troca nao travar um quadro."""
        key = (index, width, height, vivid)
        if key in self._slide_cache or (
            self._prefetch_thread is not None and self._prefetch_thread.is_alive()
        ):
            return
        self._prefetch_thread = threading.Thread(
            target=self._slide, args=(index, width, height, vivid), daemon=True
        )
        self._prefetch_thread.start()

    def _next_slide_index(self) -> int:
        """Proxima pintura numa ordem embaralhada, sem repetir a atual em seguida."""
        if not self._slide_queue:
            order = [int(index) for index in self._rng.permutation(len(self._landscape_images))]
            if len(order) > 1 and order[0] == self.landscape_index:
                order.append(order.pop(0))
            self._slide_queue = order
        return self._slide_queue.pop(0)

    def skip_slide(self) -> None:
        """Comeca ja a transicao para a proxima pintura (tecla i)."""
        self._slide_skip = True

    def _slide_state(self, now: float) -> tuple[int, int, float]:
        """(pintura atual, seguinte, fade de 0 a 1): cada uma fica VIOLA_SLIDE_SECONDS
        e passa a seguinte em VIOLA_FADE_SECONDS."""
        if self._slide_start is None:
            self._slide_start = now
        if self._slide_skip:
            self._slide_skip = False
            self._slide_start = min(self._slide_start, now - VIOLA_SLIDE_SECONDS)
        cycle = VIOLA_SLIDE_SECONDS + VIOLA_FADE_SECONDS
        while now - self._slide_start >= cycle:
            self._slide_start += cycle
            self.landscape_index = self._slide_next
            self._slide_next = self._next_slide_index()
        fade = (now - self._slide_start - VIOLA_SLIDE_SECONDS) / VIOLA_FADE_SECONDS
        return self.landscape_index, self._slide_next, min(1.0, max(0.0, fade))

    def _compose_slides(
        self, state: tuple[int, int, float], width: int, height: int, vivid: bool = False
    ) -> np.ndarray:
        """Pintura do quadro: a atual, ou o fade para a seguinte."""
        current, following, fade = state
        painting = self._slide(current, width, height, vivid)
        if fade <= 0.0:
            return painting.copy()
        return cv2.addWeighted(
            painting, 1.0 - fade, self._slide(following, width, height, vivid), fade, 0.0
        )

    def cycle_blend(self) -> None:
        """Passa a proxima mescla (tecla e); o worker aplica no proximo quadro."""
        index = VIOLA_BLENDS.index(self.viola_blend)
        self.viola_blend = VIOLA_BLENDS[(index + 1) % len(VIOLA_BLENDS)]

    def _blend_plaid(
        self,
        state: tuple[int, int, float],
        flow: Optional[np.ndarray],
        scale: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Xadrez do caipira mesclado as pinturas, conforme self.viola_blend."""
        if (
            self._field is None
            or self._field.size != (width, height)
            or self._field.layout
            != (self.plaid_layout, self.plaid_density, self.plaid_rotation)
        ):
            self._field = self._make_field(width, height)
        field = self._field
        field.gain = self.plaid_gain
        field.hue = self.plaid_hue
        field.disorder_pulses = True
        field.update(flow, scale, self.FLOW_THRESHOLD)
        now = self._viola_now
        dt = 0.0 if self._field_clock is None else min(now - self._field_clock, 0.1)
        self._field_clock = now
        field.advance(dt)
        self.line_disorder = field.disorder
        blend = self.viola_blend
        if blend == "recorte":
            painting = self._compose_slides(state, width, height)
            shift = self._warp_shift(field, width, height)
            if shift is None:
                return painting
            mask = np.zeros((height, width), dtype=np.uint8)
            field.draw_mask(mask)
            cv2.copyTo(self._apply_warp(painting, shift), mask, painting)
            return painting
        if blend in ("tecido", "quadros"):
            shift = self._warp_shift(field, width, height)
            if blend == "tecido":
                painting = self._compose_slides(state, width, height)
            else:
                shift = self._lag_shift(field, shift, width, height)
                painting = self._mosaic_slides(field, state, width, height)
            body = cv2.convertScaleAbs(
                self._apply_warp(painting, shift), alpha=VIOLA_TEXTILE_BRIGHTNESS
            )
            field.draw(body)
            return body
        # Janela e cores: as linhas (e os pulsos) viram a mascara de luz da pintura.
        lines = np.zeros((height, width, 3), dtype=np.uint8)
        field.draw(lines)
        light = cv2.cvtColor(lines, cv2.COLOR_BGR2GRAY)
        if blend == "cores":
            # Linhas finas e tons escuros da paleta tambem acendem por inteiro.
            light = cv2.convertScaleAbs(light, alpha=VIOLA_COLOR_LIGHT_GAIN)
            painting = self._compose_slides(state, width, height, vivid=True)
            background = VIOLA_COLOR_BACKGROUND
        else:
            painting = self._compose_slides(state, width, height)
            background = VIOLA_WINDOW_BACKGROUND
        lit = cv2.multiply(painting, cv2.cvtColor(light, cv2.COLOR_GRAY2BGR), scale=1.0 / 255.0)
        return cv2.max(cv2.convertScaleAbs(painting, alpha=background), lit)

    def _warp_shift(
        self, field: PlaidStack, width: int, height: int
    ) -> Optional[np.ndarray]:
        """Deslocamento (grade, grade, 2) das linhas em celulas de VIOLA_WARP_CELL px,
        ou None com a malha em repouso."""
        positions, moves = field.node_motion()
        if len(moves) == 0 or float(np.abs(moves).max()) < 0.05:
            return None
        cell = VIOLA_WARP_CELL
        grid_width, grid_height = width // cell + 1, height // cell + 1
        columns = np.clip((positions[:, 0] / cell).astype(np.intp), 0, grid_width - 1)
        rows = np.clip((positions[:, 1] / cell).astype(np.intp), 0, grid_height - 1)
        flat = rows * grid_width + columns
        cells = grid_width * grid_height
        # Soma dos deslocamentos e numero de nos por celula, espalhados pelo blur.
        layers = [
            np.bincount(flat, weights=weights, minlength=cells)
            .reshape(grid_height, grid_width)
            .astype(np.float32)
            for weights in (moves[:, 0], moves[:, 1], None)
        ]
        spread = cv2.GaussianBlur(
            np.dstack(layers), (0, 0), VIOLA_WARP_SIGMA, borderType=cv2.BORDER_REPLICATE
        )
        return spread[..., :2] / (spread[..., 2:] + 1e-3)

    def _apply_warp(self, painting: np.ndarray, shift: Optional[np.ndarray]) -> np.ndarray:
        """Deforma a pintura com o deslocamento da grade, num so remap."""
        if shift is None:
            return painting
        height, width = painting.shape[:2]
        grid_height, grid_width = shift.shape[:2]
        # Mapa inverso montado na grade pequena e ampliado num so resize: cada pixel
        # busca a cor de onde a linha veio. O resize bilinear reproduz exatamente as
        # coordenadas da grade (lineares), sem somas em resolucao cheia.
        if self._warp_size != (width, height, grid_width, grid_height):
            centers_x = (np.arange(grid_width, dtype=np.float32) + 0.5) * (width / grid_width) - 0.5
            centers_y = (
                (np.arange(grid_height, dtype=np.float32) + 0.5) * (height / grid_height) - 0.5
            )
            self._warp_grid = np.dstack(np.meshgrid(centers_x, centers_y))
            self._warp_size = (width, height, grid_width, grid_height)
        warp_map = cv2.resize(
            self._warp_grid - shift, (width, height), interpolation=cv2.INTER_LINEAR
        )
        return cv2.remap(painting, warp_map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _mosaic_cells(
        self, field: PlaidStack, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quadrado do xadrez de cada celula da grade e de cada pixel (0 a 255).

        Calculado uma vez por malha (~ dezenas de ms em 720p), na malha mais proxima,
        em repouso; quadrados distantes podem repetir o mesmo numero.
        """
        key = (id(field), width, height)
        if self._mosaic_key == key:
            return self._mosaic_maps
        layer = field._layers[-1][0]
        axes = []
        for family in layer._families:
            if len(family) == 0:
                continue
            rest = family._rest
            direction = rest[0, -1] - rest[0, 0]
            direction = direction / max(1e-6, float(np.hypot(*direction)))
            normal = np.array((-direction[1], direction[0]), dtype=np.float32)
            axes.append((normal, np.sort(rest[:, 0] @ normal)))

        def cells_at(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            code = np.zeros(x.shape, dtype=np.int64)
            for factor, (normal, offsets) in zip((73856093, 19349663), axes):
                band = np.searchsorted(offsets, x * normal[0] + y * normal[1])
                code ^= band.astype(np.int64) * factor
            return (code & 255).astype(np.uint8)

        cell = VIOLA_WARP_CELL
        grid_width, grid_height = width // cell + 1, height // cell + 1
        grid_x, grid_y = np.meshgrid(
            (np.arange(grid_width, dtype=np.float32) + 0.5) * (width / grid_width) - 0.5,
            (np.arange(grid_height, dtype=np.float32) + 0.5) * (height / grid_height) - 0.5,
        )
        pixel_x, pixel_y = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        self._mosaic_maps = (cells_at(grid_x, grid_y), cells_at(pixel_x, pixel_y))
        self._mosaic_key = key
        self._mosaic_shift = None
        return self._mosaic_maps

    def _lag_shift(
        self, field: PlaidStack, shift: Optional[np.ndarray], width: int, height: int
    ) -> Optional[np.ndarray]:
        """Quadros: cada quadrado segue o deslocamento com o seu proprio atraso."""
        cells, _ = self._mosaic_cells(field, width, height)
        target = np.zeros((*cells.shape, 2), np.float32) if shift is None else shift
        if self._mosaic_shift is None or self._mosaic_shift.shape != target.shape:
            self._mosaic_shift = np.zeros_like(target)
        lagged = self._mosaic_shift
        lagged += self._mosaic_rates[cells][..., None] * (target - lagged)
        if float(np.abs(lagged).max()) < 0.05:
            return None
        return lagged

    def _mosaic_slides(
        self, field: PlaidStack, state: tuple[int, int, float], width: int, height: int
    ) -> np.ndarray:
        """Quadros: na troca, cada quadrado passa a pintura seguinte no seu momento."""
        current, following, fade = state
        if fade <= 0.0:
            return self._slide(current, width, height)
        if fade >= 1.0:
            return self._slide(following, width, height)
        _, pixels = self._mosaic_cells(field, width, height)
        # Cada quadrado faz o seu fade em (1 - VIOLA_MOSAIC_STAGGER) do tempo, comecando
        # num atraso sorteado; so nos quadros da troca.
        weights = np.clip(
            (fade - self._mosaic_offsets) / (1.0 - VIOLA_MOSAIC_STAGGER), 0.0, 1.0
        )
        alpha = cv2.LUT(pixels, (weights * 255.0).astype(np.uint8)).astype(np.float32)
        alpha *= 1.0 / 255.0
        return cv2.blendLinear(
            self._slide(current, width, height),
            self._slide(following, width, height),
            1.0 - alpha,
            alpha,
        )

    def _process_role(
        self, frame_bgr: np.ndarray, gray: np.ndarray, body_mask: Optional[np.ndarray] = None
    ) -> VisionResult:
        """Roda de viola: silhuetas recortadas em paisagens sobre branco, com arpejos."""
        height, width = frame_bgr.shape[:2]
        flow = None if self._previous_gray is None else self._farneback(gray)
        scale = np.array((gray.shape[1] / width, gray.shape[0] / height), dtype=np.float32)
        boxes, masks = self._segmenter.people(frame_bgr)
        self._people_source = "YOLO-seg"
        self._people_history.append(min(len(boxes), CAIPIRA_MAX_PEOPLE))
        self.people_count = sorted(self._people_history)[len(self._people_history) // 2]
        self._people_yolo_boxes = boxes
        self._mix_arpeggios(boxes, flow, scale, (width, height))
        now = self._viola_now
        self._assign_paintings()
        self._update_silhouette_masks(boxes, masks, now, width, height)
        body = self._role_canvas(width, height)
        for player in self._viola_players:
            self._paste_silhouette(body, player)
        self._draw_strings(body, now)
        debug = frame_bgr.copy()
        self._draw_people_debug(debug)
        return VisionResult(None, debug, body)

    def _update_silhouette_masks(
        self,
        boxes: list[tuple[int, int, int, int]],
        masks: list[np.ndarray],
        now: float,
        width: int,
        height: int,
    ) -> None:
        """Mascara de cada pessoa rastreada, em ROLE_ALPHA_SCALE, suavizada no tempo;
        quem nao foi visto neste quadro vai apagando."""
        small_size = (
            max(1, round(width * ROLE_ALPHA_SCALE)),
            max(1, round(height * ROLE_ALPHA_SCALE)),
        )
        detections = {box: mask for box, mask in zip(boxes, masks)}
        for player in self._viola_players:
            mask = detections.get(player.box) if player.seen == now else None
            target = (
                np.zeros(small_size[::-1], np.float32)
                if mask is None
                else cv2.resize(
                    mask,
                    small_size,
                    interpolation=cv2.INTER_LINEAR
                    if self.role_edge == "suave"
                    else cv2.INTER_NEAREST,
                )
            )
            if player.alpha is None or player.alpha.shape != target.shape:
                player.alpha = target
            else:
                player.alpha += ROLE_MASK_SMOOTHING * (target - player.alpha)

    def _string_anchors(self, player: ViolaPlayer) -> np.ndarray:
        """ROLE_STRING_COUNT pontos sorteados dentro da silhueta, relativos a caixa."""
        x0, y0, x1, y1 = player.box
        size = np.array((max(1, x1 - x0), max(1, y1 - y0)), dtype=np.float32)
        if player.alpha is not None:
            rows, columns = np.nonzero(player.alpha >= 0.5)
            points = (np.stack((columns, rows), axis=1).astype(np.float32) + 0.5) / ROLE_ALPHA_SCALE
            inside = np.all((points >= (x0, y0)) & (points <= (x1, y1)), axis=1)
            points = points[inside]
            if len(points) >= ROLE_STRING_COUNT:
                chosen = points[self._rng.choice(len(points), ROLE_STRING_COUNT, replace=False)]
                return (chosen - (x0, y0)) / size
        return self._rng.uniform(0.2, 0.8, (ROLE_STRING_COUNT, 2)).astype(np.float32)

    def _draw_strings(self, body: np.ndarray, now: float) -> None:
        """Liga cada pessoa a cada outra por cordas finas, com pontas espalhadas pelas
        silhuetas, que vibram mais quando os corpos estao perto."""
        players = [
            player
            for player in self._viola_players
            if now - player.seen <= ROLE_STRING_HOLD
        ]
        anchors_by_pair = {}
        if len(players) < 2:
            self._string_pairs = anchors_by_pair
            return
        height, width = body.shape[:2]
        diagonal = math.hypot(width, height)
        t = np.linspace(0.0, 1.0, ROLE_STRING_SAMPLES, dtype=np.float32)[None, :, None]
        slow, fast = ROLE_STRING_HZ
        # Modo de cada corda (metade fundamental, metade oitava) e fase propria.
        harmonics = np.where(np.arange(ROLE_STRING_COUNT) % 2 == 0, 1.0, 2.0).astype(np.float32)
        strings = []
        for first in range(len(players)):
            for second in range(first + 1, len(players)):
                one, other = players[first], players[second]
                key = (id(one), id(other))
                anchors = self._string_pairs.get(key)
                if anchors is None:
                    anchors = (self._string_anchors(one), self._string_anchors(other))
                anchors_by_pair[key] = anchors
                ends = []
                for player, relative in zip((one, other), anchors):
                    x0, y0, x1, y1 = player.box
                    ends.append(
                        np.array((x0, y0), np.float32)
                        + relative * np.array((max(1, x1 - x0), max(1, y1 - y0)), np.float32)
                    )
                start, end = ends
                along = end - start
                lengths = np.maximum(np.hypot(along[:, 0], along[:, 1]), 1.0)[:, None]
                normals = np.stack((-along[:, 1], along[:, 0]), axis=1) / lengths
                # Proximidade pelos centros dos corpos: o feixe todo vibra junto.
                gap = math.hypot(one.center[0] - other.center[0], one.center[1] - other.center[1])
                closeness = min(
                    1.0,
                    max(0.0, (ROLE_STRING_FAR - gap / diagonal) / (ROLE_STRING_FAR - ROLE_STRING_NEAR)),
                )
                amplitude = ROLE_STRING_AMPLITUDE * height * (
                    ROLE_STRING_IDLE + (1.0 - ROLE_STRING_IDLE) * closeness
                )
                frequency = slow + (fast - slow) * closeness
                phases = (one.slot * 7 + other.slot * 13 + np.arange(ROLE_STRING_COUNT) * 1.3) % (
                    2.0 * math.pi
                )
                swing = np.sin(2.0 * math.pi * frequency * harmonics * now + phases)
                wave = (
                    (amplitude / harmonics * swing)[:, None, None]
                    * np.sin(math.pi * harmonics[:, None, None] * t)
                    * normals[:, None, :]
                )
                curves = start[:, None, :] + along[:, None, :] * t + wave
                strings.extend(np.rint(curves).astype(np.int32))
        self._string_pairs = anchors_by_pair
        if not strings:
            return
        # Fios de 1 px como tinta semitransparente, so na regiao das cordas.
        points = np.concatenate(strings)
        x0, y0 = np.maximum(0, points.min(axis=0) - 2)
        x1, y1 = np.minimum((width, height), points.max(axis=0) + 3)
        if x1 <= x0 or y1 <= y0:
            return
        ink = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.polylines(ink, [curve - (x0, y0) for curve in strings], False, 255, 1, cv2.LINE_AA)
        keep = cv2.convertScaleAbs(ink, alpha=-ROLE_STRING_OPACITY, beta=255.0)
        region = body[y0:y1, x0:x1]
        body[y0:y1, x0:x1] = cv2.multiply(
            region, cv2.cvtColor(keep, cv2.COLOR_GRAY2BGR), scale=1.0 / 255.0
        )

    def _role_canvas(self, width: int, height: int) -> np.ndarray:
        """Fundo branco do quadro (copiado de um guardado)."""
        if self._white is None or self._white.shape[:2] != (height, width):
            self._white = np.full((height, width, 3), 255, np.uint8)
        return self._white.copy()

    def cycle_role_edge(self) -> None:
        """Passa ao proximo estilo de borda das silhuetas (tecla e)."""
        index = ROLE_EDGE_STYLES.index(self.role_edge)
        self.role_edge = ROLE_EDGE_STYLES[(index + 1) % len(ROLE_EDGE_STYLES)]

    def _paste_silhouette(self, body: np.ndarray, player: ViolaPlayer) -> None:
        """Compoe a silhueta da pessoa, so dentro da caixa dela: a paisagem dela (roda)
        ou o xadrez dela, com os vaos transparentes (jeca)."""
        height, width = body.shape[:2]
        region = self._silhouette_region(player, width, height)
        if region is None:
            return
        x0, y0, x1, y1, cover = region
        if self.mode == JECA_MODE:
            source, pattern = self._plaid_patch(player, x0, y0, x1, y1, width, height)
            if cover.dtype == bool:
                cover = cover & pattern
            else:
                cover = cover * pattern
        else:
            if player.painting is None:
                return
            source = self._slide(player.painting, width, height)[y0:y1, x0:x1]
        if cover.dtype == bool:
            # cv2.copyTo grava direto na regiao (~50x mais rapido que np.copyto com where).
            cv2.copyTo(source, cover.view(np.uint8), body[y0:y1, x0:x1])
        else:
            body[y0:y1, x0:x1] = cv2.blendLinear(body[y0:y1, x0:x1], source, 1.0 - cover, cover)

    def _silhouette_region(
        self, player: ViolaPlayer, width: int, height: int
    ) -> Optional[tuple[int, int, int, int, np.ndarray]]:
        """(x0, y0, x1, y1, cobertura) da silhueta no quadro: cobertura booleana nas
        bordas duras (poligono, serrilhado) ou peso de 0 a 1 na suave."""
        if player.alpha is None:
            return None
        style = self.role_edge
        if style == "suave":
            alpha = cv2.GaussianBlur(player.alpha, (0, 0), ROLE_EDGE_SIGMA * ROLE_ALPHA_SCALE)
        else:
            alpha = (player.alpha >= 0.5).astype(np.uint8)
        if style == "poligono":
            return self._polygon_region(alpha, player, width, height)
        x, y, box_width, box_height = cv2.boundingRect((alpha > 0.01).astype(np.uint8))
        if box_width == 0 or box_height == 0:
            return None
        x0 = int(x / ROLE_ALPHA_SCALE)
        y0 = int(y / ROLE_ALPHA_SCALE)
        x1 = min(width, int(math.ceil((x + box_width) / ROLE_ALPHA_SCALE)))
        y1 = min(height, int(math.ceil((y + box_height) / ROLE_ALPHA_SCALE)))
        crop = alpha[y : y + box_height, x : x + box_width]
        if style == "serrilhado":
            cover = cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
            return x0, y0, x1, y1, cover.astype(bool)
        weight = cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        np.clip(weight, 0.0, 1.0, out=weight)
        return x0, y0, x1, y1, weight

    def _polygon_region(
        self, alpha: np.ndarray, player: ViolaPlayer, width: int, height: int
    ) -> Optional[tuple[int, int, int, int, np.ndarray]]:
        """Poligono: contorno externo simplificado em poucos lados retos e preenchido.

        O poligono de cada pessoa e mantido entre quadros: enquanto a forma nova so
        difere da atual pelo ruido da mascara ou por um deslocamento, as facetas
        ficam e so acompanham o corpo; elas so sao refeitas quando a pose muda.
        """
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Tolerancia e area minima na escala da mascara pequena.
        epsilon = ROLE_POLYGON_EPSILON * alpha.shape[0]
        min_area = ROLE_POLYGON_MIN_AREA * alpha.size
        fresh = [
            cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float32)
            for contour in contours
            if cv2.contourArea(contour) >= min_area
        ]
        if not fresh:
            player.polygons = None
            return None
        # Centro da mascara (ja suavizada no tempo): referencia de posicao menos ruidosa
        # que o centro do poligono, que muda com a escolha dos vertices.
        moments = cv2.moments(alpha, True)
        if moments["m00"] == 0:
            player.polygons = None
            return None
        center = np.array(
            (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]), dtype=np.float32
        )
        polygons = fresh
        if player.polygons is not None:
            # Zona morta continua: sob o ruido a posicao nao muda; acima dela, as
            # facetas seguem o corpo continuamente, atrasadas so pela margem.
            drift = center - player.polygon_center
            distance = float(np.hypot(*drift))
            if distance > ROLE_POLYGON_DEADBAND:
                step = drift * (1.0 - ROLE_POLYGON_DEADBAND / distance)
                player.polygons = [polygon + step for polygon in player.polygons]
                player.polygon_center = player.polygon_center + step
            held = player.polygons
            fresh_area = self._raster_polygons(fresh, alpha.shape)
            held_area = self._raster_polygons(held, alpha.shape)
            union = cv2.countNonZero(cv2.bitwise_or(fresh_area, held_area))
            overlap = cv2.countNonZero(cv2.bitwise_and(fresh_area, held_area))
            if union and overlap / union >= ROLE_POLYGON_HOLD_IOU:
                polygons = held
        if polygons is fresh:
            player.polygon_center = center
        player.polygons = polygons
        polygons = [(polygon + 0.5) / ROLE_ALPHA_SCALE for polygon in polygons]
        points = np.concatenate(polygons)
        x0, y0 = np.maximum(0, np.floor(points.min(axis=0)).astype(int))
        x1, y1 = np.minimum((width, height), np.ceil(points.max(axis=0)).astype(int) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        inside = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(
            inside,
            [np.round(polygon - (x0, y0)).astype(np.int32) for polygon in polygons],
            1,
            cv2.LINE_8,
        )
        return int(x0), int(y0), int(x1), int(y1), inside.astype(bool)

    def _plaid_patch(
        self,
        player: ViolaPlayer,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cores e linhas (booleano) do xadrez da pessoa na regiao dada.

        O tecido, girado no angulo fixo da pessoa, e desenhado uma vez do tamanho do
        quadro mais uma margem de um periodo na diagonal. O xadrez acompanha o corpo
        como uma roupa: o deslocamento e reduzido a um periodo nos eixos do padrao
        (que se repete neles), e a regiao e um recorte do tecido, sem copia.
        """
        if player.plaid is None or player.plaid[4] != (width, height):
            period, stripes, palette = make_jeca_plaid(
                self._rng, JECA_PLAID_HUES[(player.slot - 1) % len(JECA_PLAID_HUES)], height
            )
            lattice, angle = jeca_plaid_lattice(period, float(self._rng.uniform(0.0, 180.0)))
            margin = math.ceil(float(np.hypot(*lattice)) * math.sqrt(2.0)) + 2
            colors, lines = render_jeca_plaid(
                period, stripes, palette, lattice, width + 2 * margin, height + 2 * margin
            )
            player.plaid = (colors, lines, lattice, margin, (width, height), angle)
            player.plaid_origin = np.array(player.center, dtype=np.float32)
            player.plaid_offset = np.zeros(2, dtype=np.float32)
        colors, lines, lattice, margin, _, _ = player.plaid
        # Deslocamento do corpo desde que o xadrez nasceu, com zona morta.
        drift = np.array(player.center, dtype=np.float32) - player.plaid_origin - player.plaid_offset
        distance = float(np.hypot(*drift))
        if distance > JECA_PLAID_DEADBAND:
            player.plaid_offset += drift * (1.0 - JECA_PLAID_DEADBAND / distance)
        # Reduz o deslocamento (inteiro) a uma celula da rede do padrao, subtraindo
        # multiplos inteiros dos vetores (a, b) e (-b, a): o resto tem |resto| <= passo
        # * raiz 2 e o recorte e exatamente o mesmo.
        offset = np.rint(player.plaid_offset).astype(int)
        a, b = int(lattice[0]), int(lattice[1])
        norm = a * a + b * b
        along = math.floor((offset[0] * a + offset[1] * b) / norm)
        across = math.floor((-offset[0] * b + offset[1] * a) / norm)
        reduced = offset - along * np.array((a, b)) - across * np.array((-b, a))
        shift_x, shift_y = margin - reduced
        rows = slice(y0 + shift_y, y1 + shift_y)
        columns = slice(x0 + shift_x, x1 + shift_x)
        return colors[rows, columns], lines[rows, columns]

    def _assign_paintings(self) -> None:
        """Paisagem de cada pessoa nova: sorteada entre as que ninguem usa, e fixa."""
        used = {player.painting for player in self._viola_players}
        for player in self._viola_players:
            if player.painting is None:
                free = [
                    index for index in range(len(self._landscape_images)) if index not in used
                ] or list(range(len(self._landscape_images)))
                player.painting = free[int(self._rng.integers(len(free)))]
                used.add(player.painting)

    def _extremities(self, frame_bgr: np.ndarray, now: float) -> list[tuple[str, np.ndarray, int]]:
        """Cabeca, maos e pes de cada pessoa (pontos confiaveis do YOLO de pose), com a
        silhueta (slot) a que pertencem.

        Uma pose so conta se o centro dos seus pontos cair dentro da caixa de uma
        silhueta, e cada silhueta fica com uma pose so, a mais confiante; poses com
        confianca media abaixo de JECATATU_POSE_CONFIDENCE (fantasmas do fundo) saem.
        """
        if self._yolo is None:
            return []
        points, confidence = self._yolo_keypoints(frame_bgr)
        players = [player for player in self._viola_players if player.seen == now]
        chosen: dict[int, tuple[float, int]] = {}
        for index, person in enumerate(points[:YOLO_MAX_DETECTIONS]):
            score = 1.0 if confidence is None else float(confidence[index].mean())
            reliable = (
                person
                if confidence is None
                else person[confidence[index] >= YOLO_MIN_KEYPOINT_CONFIDENCE]
            )
            if score < JECATATU_POSE_CONFIDENCE or len(reliable) == 0:
                continue
            center = reliable.mean(axis=0)
            inside = [
                player
                for player in players
                if player.box[0] <= center[0] <= player.box[2]
                and player.box[1] <= center[1] <= player.box[3]
            ]
            if not inside:
                continue
            slot = min(
                inside,
                key=lambda player: math.hypot(
                    center[0] - player.center[0], center[1] - player.center[1]
                ),
            ).slot
            if slot not in chosen or score > chosen[slot][0]:
                chosen[slot] = (score, index)
        found = []
        for slot, (_, index) in chosen.items():
            person = points[index]
            for part, keypoints in JECATATU_KEYPOINTS:
                keypoints = list(keypoints)
                if confidence is not None:
                    keypoints = [
                        k for k in keypoints if confidence[index, k] >= YOLO_MIN_KEYPOINT_CONFIDENCE
                    ]
                if keypoints:
                    found.append((part, person[keypoints].mean(axis=0).astype(np.float32), slot))
        return found

    def _update_contacts(
        self, extremities: list[tuple[str, np.ndarray, int]], now: float, width: int, height: int
    ) -> list[dict]:
        """Liga cada ponto ao do quadro anterior (mesma parte, o mais perto) e devolve
        os contatos: gestos rapidos, com rearme e intervalo minimo."""
        previous = self._contacts
        max_jump = JECATATU_CONTACT_JUMP * math.hypot(width, height)
        unit = height / 720.0
        used: set[int] = set()
        current = []
        for part, point, slot in extremities:
            best, best_distance = None, max_jump
            for index, contact in enumerate(previous):
                if index in used or contact["part"] != part or contact["slot"] != slot:
                    continue
                distance = float(np.hypot(*(point - contact["point"])))
                if distance < best_distance:
                    best, best_distance = index, distance
            if best is None:
                current.append(
                    {"part": part, "slot": slot, "point": point, "steps": [],
                     "velocity": np.zeros(2, np.float32), "armed": True, "last": -math.inf}
                )
            else:
                used.add(best)
                old = previous[best]
                # Velocidade robusta: mediana dos ultimos passos, que descarta um salto
                # isolado do modelo de pose; um gesto brusco de verdade dura mais.
                steps = (old["steps"] + [point - old["point"]])[-JECATATU_VELOCITY_FRAMES:]
                current.append(
                    {"part": part, "slot": slot, "point": point, "steps": steps,
                     "velocity": np.median(np.array(steps), axis=0).astype(np.float32),
                     "armed": old["armed"], "last": old["last"]}
                )
        events = []
        for contact in current:
            speed = float(np.hypot(*contact["velocity"])) / unit
            contact["speed"] = speed
            if speed < JECATATU_CONTACT_RELEASE:
                contact["armed"] = True
            elif (
                contact["armed"]
                and speed >= JECATATU_CONTACT_SPEED
                and now - contact["last"] >= JECATATU_CONTACT_COOLDOWN
            ):
                contact["armed"] = False
                contact["last"] = now
                events.append(contact)
        self._contacts = current
        return events

    def _contact_flow(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Campo de fluxo sintetico (grade do optical flow) com a velocidade de cada
        ponto de contato espalhada num raio: e o que empurra as linhas."""
        grid_width = max(1, int(width * self.FLOW_SCALE))
        grid_height = max(1, int(height * self.FLOW_SCALE))
        flow = np.zeros((grid_height, grid_width, 2), dtype=np.float32)
        radius = max(1, round(JECATATU_PUSH_RADIUS * height * self.FLOW_SCALE))
        unit = height / 720.0
        for contact in self._contacts:
            velocity = contact["velocity"]
            if float(np.hypot(*velocity)) < JECATATU_PUSH_MIN * unit:
                continue
            push = velocity * (self.FLOW_SCALE * JECATATU_PUSH_GAIN)
            center = tuple(int(round(value * self.FLOW_SCALE)) for value in contact["point"])
            cv2.circle(flow, center, radius, (float(push[0]), float(push[1])), -1)
        if radius > 1:
            flow = cv2.GaussianBlur(flow, (0, 0), radius / 2.0)
        scale = np.array((grid_width / width, grid_height / height), dtype=np.float32)
        return flow, scale

    def _process_jecatatu(
        self, frame_bgr: np.ndarray, body_mask: Optional[np.ndarray] = None
    ) -> VisionResult:
        """Jecatatu: silhuetas em paisagens sobre preto, xadrez por cima reagindo a
        cabeca, maos e pes; cada contato faz a silhueta brilhar. Cada pessoa toca o seu
        arpejo no seu tempo; silhuetas fundidas sincronizam."""
        height, width = frame_bgr.shape[:2]
        boxes, masks = self._segmenter.people(frame_bgr)
        self._people_source = "YOLO-seg"
        self._people_history.append(min(len(boxes), CAIPIRA_MAX_PEOPLE))
        self.people_count = sorted(self._people_history)[len(self._people_history) // 2]
        self._people_yolo_boxes = boxes
        now = self._track_players(
            boxes, None, np.ones(2, np.float32), (width, height),
            list(range(1, CAIPIRA_MAX_PEOPLE + 1)), False,
        )
        self._assign_paintings()
        self._update_silhouette_masks(boxes, masks, now, width, height)
        events = self._update_contacts(self._extremities(frame_bgr, now), now, width, height)
        if (
            self._field is None
            or self._field.size != (width, height)
            or self._field.layout != (self.plaid_layout, self.plaid_density, self.plaid_rotation)
        ):
            self._field = self._make_field(width, height)
        field = self._field
        field.gain = self.plaid_gain
        field.hue = self.plaid_hue
        flow, scale = self._contact_flow(width, height)
        field.update(flow, scale, self.FLOW_THRESHOLD)
        unit = height / 720.0
        lit = {contact["slot"] for contact in events}
        self.contact_count += len(events)
        dt = 0.0 if self._field_clock is None else min(now - self._field_clock, 0.1)
        self._field_clock = now
        fade = math.exp(-dt / JECATATU_GLOW_SECONDS)
        for player in self._viola_players:
            player.glow = 1.0 if player.slot in lit else player.glow * fade
        if self._black is None or self._black.shape[:2] != (height, width):
            self._black = np.full((height, width, 3), JECATATU_BACKGROUND, np.uint8)
        body = self._black.copy()
        groups = self._silhouette_groups(height)
        for group in groups:
            if len(group) == 1:
                self._paste_silhouette(body, group[0])
            else:
                self._paste_group(body, group)
        self._play_group_arpeggios(groups)
        field.draw(body)
        for player in self._viola_players:
            if player.glow > 0.02:
                self._draw_glow(body, player)
        debug = frame_bgr.copy()
        self._draw_people_debug(debug)
        for contact in self._contacts:
            point = tuple(int(round(value)) for value in contact["point"])
            color = JECATATU_PART_COLORS[contact["part"]]
            cv2.circle(debug, point, max(4, round(6 * unit)), color, -1, cv2.LINE_AA)
            if now - contact["last"] < 0.25:
                cv2.circle(debug, point, max(10, round(22 * unit)), color, 3, cv2.LINE_AA)
        return VisionResult(None, debug, body)

    def _draw_glow(self, body: np.ndarray, player: ViolaPlayer) -> None:
        """Brilho que sai da silhueta: a mascara desfocada e colorida, somada por cima de
        tudo, so na regiao da pessoa. Calculado em 8 bits e em JECATATU_GLOW_SCALE da
        mascara pequena (o brilho e difuso), e so ampliado para compor."""
        if player.alpha is None:
            return
        height, width = body.shape[:2]
        mask = (player.alpha >= 0.5).astype(np.uint8) * 255
        x, y, box_width, box_height = cv2.boundingRect(mask)
        if box_width == 0 or box_height == 0:
            return
        sigma = max(1.0, JECATATU_GLOW_RADIUS * height * ROLE_ALPHA_SCALE)
        reach = int(math.ceil(3 * sigma))
        sx0, sy0 = max(0, x - reach), max(0, y - reach)
        sx1 = min(mask.shape[1], x + box_width + reach)
        sy1 = min(mask.shape[0], y + box_height + reach)
        crop = mask[sy0:sy1, sx0:sx1]
        small = cv2.resize(
            crop,
            (max(1, round(crop.shape[1] * JECATATU_GLOW_SCALE)), max(1, round(crop.shape[0] * JECATATU_GLOW_SCALE))),
            interpolation=cv2.INTER_AREA,
        )
        halo = cv2.GaussianBlur(small, (0, 0), max(0.8, sigma * JECATATU_GLOW_SCALE))
        # A luz sai do corpo: forte em volta, so JECATATU_GLOW_INSIDE dela por dentro,
        # para clarear a paisagem sem apaga-la.
        halo = cv2.subtract(
            halo, cv2.multiply(halo, small, scale=(1.0 - JECATATU_GLOW_INSIDE) / 255.0)
        )
        pure = np.array(plaid_palette(self.plaid_hue)[0], dtype=np.float64)
        color = (pure + JECATATU_GLOW_WHITE * (255.0 - pure)) * (JECATATU_GLOW_GAIN * player.glow)
        light = cv2.multiply(
            cv2.cvtColor(halo, cv2.COLOR_GRAY2BGR), (*color.tolist(), 0.0), scale=1.0 / 255.0
        )
        x0, y0 = int(sx0 / ROLE_ALPHA_SCALE), int(sy0 / ROLE_ALPHA_SCALE)
        x1 = min(width, int(math.ceil(sx1 / ROLE_ALPHA_SCALE)))
        y1 = min(height, int(math.ceil(sy1 / ROLE_ALPHA_SCALE)))
        light = cv2.resize(light, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        region = body[y0:y1, x0:x1]
        cv2.add(region, light, dst=region)

    def _play_group_arpeggios(self, groups: list[list[ViolaPlayer]]) -> None:
        """Cada pessoa toca o seu canal; um grupo fundido segue o tempo do menor canal."""
        if self.arpeggios is None:
            return
        present = set()
        for group in groups:
            leader = min(player.slot for player in group)
            for player in group:
                present.add(player.slot)
                self.arpeggios.follow(player.slot, leader)
        for channel in self.arpeggios.channels:
            self.arpeggios.set_volume(
                channel, JECATATU_ARPEGGIO_VOLUME if channel in present else 0.0
            )

    def _arpeggio_label(self) -> str:
        """Canais tocando, juntando os que estao no mesmo tempo (ex.: 1+3 2)."""
        if self.arpeggios is None:
            return "sem arpejos"
        tempos: dict[int, list[int]] = {}
        for channel, volume in sorted(self.arpeggios.volumes.items()):
            if volume > 0:
                tempos.setdefault(self.arpeggios.leader_of(channel), []).append(channel)
        if not tempos:
            return "arpejos mudos"
        return "arpejos " + " ".join(
            "+".join(str(channel) for channel in channels) for channels in tempos.values()
        )

    def _silhouette_groups(self, height: int) -> list[list[ViolaPlayer]]:
        """Pessoas agrupadas pelas silhuetas que se tocam (com folga para separar)."""
        players = [player for player in self._viola_players if player.alpha is not None]
        masks = [(player.alpha >= 0.5).astype(np.uint8) for player in players]
        boxes = [cv2.boundingRect(mask) for mask in masks]
        near = JECATATU_MERGE_DISTANCE * height * ROLE_ALPHA_SCALE
        far = JECATATU_SPLIT_DISTANCE * height * ROLE_ALPHA_SCALE
        merged = set()
        for first in range(len(players)):
            for second in range(first + 1, len(players)):
                pair = tuple(sorted((players[first].slot, players[second].slot)))
                limit = far if pair in self._merged_pairs else near
                gap = self._mask_gap(masks[first], boxes[first], masks[second], boxes[second], limit)
                if gap is not None and gap <= limit:
                    merged.add(pair)
        self._merged_pairs = merged
        # Componentes conexos dos pares somados (union-find pelos slots).
        parent = {player.slot: player.slot for player in players}

        def root(slot: int) -> int:
            while parent[slot] != slot:
                parent[slot] = parent[parent[slot]]
                slot = parent[slot]
            return slot

        for first, second in merged:
            parent[root(first)] = root(second)
        groups: dict[int, list[ViolaPlayer]] = {}
        for player in players:
            groups.setdefault(root(player.slot), []).append(player)
        # Paisagem de cada grupo novo: uma que nenhuma pessoa nem grupo visivel usa.
        current = {frozenset(player.slot for player in group) for group in groups.values() if len(group) > 1}
        self._group_paintings = {
            key: painting for key, painting in self._group_paintings.items() if key in current
        }
        self._group_shapes = {
            key: shape for key, shape in self._group_shapes.items() if key in current
        }
        for key in current:
            if key not in self._group_paintings:
                used = {player.painting for player in players} | set(self._group_paintings.values())
                free = [
                    index for index in range(len(self._landscape_images)) if index not in used
                ] or list(range(len(self._landscape_images)))
                self._group_paintings[key] = free[int(self._rng.integers(len(free)))]
        return list(groups.values())

    @staticmethod
    def _mask_gap(
        first: np.ndarray,
        first_box: tuple[int, int, int, int],
        second: np.ndarray,
        second_box: tuple[int, int, int, int],
        limit: float,
    ) -> Optional[float]:
        """Menor distancia (px da mascara pequena) entre duas mascaras, se for ate limit;
        None se as caixas, alargadas por limit, nem se encostam. So a regiao entre as
        duas e examinada."""
        reach = int(math.ceil(limit)) + 1
        x0 = max(first_box[0], second_box[0]) - reach
        y0 = max(first_box[1], second_box[1]) - reach
        x1 = min(first_box[0] + first_box[2], second_box[0] + second_box[2]) + reach
        y1 = min(first_box[1] + first_box[3], second_box[1] + second_box[3]) + reach
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(first.shape[1], x1), min(first.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return None
        first_crop = first[y0:y1, x0:x1]
        second_crop = second[y0:y1, x0:x1]
        if not first_crop.any() or not second_crop.any():
            return None
        # Distancia de cada pixel ao pixel mais proximo da primeira mascara.
        distance = cv2.distanceTransform((first_crop == 0).astype(np.uint8), cv2.DIST_L2, 3)
        return float(distance[second_crop > 0].min())

    def _paste_group(self, body: np.ndarray, group: list[ViolaPlayer]) -> None:
        """Silhuetas somadas numa forma so, preenchida pela paisagem do grupo.

        As mascaras das pessoas sao unidas e fechadas (dilata e erode, MORPH_CLOSE):
        os vaos mais estreitos que a distancia de separacao viram uma ponte entre os
        corpos. O resultado vira um poligono so, estabilizado como os das pessoas;
        e uma fusao aproximada, que tambem preenche reentrancias pequenas da forma.
        """
        height, width = body.shape[:2]
        masks = [player.alpha for player in group if player.alpha is not None]
        if not masks:
            return
        union = np.zeros_like(masks[0], dtype=np.uint8)
        for alpha in masks:
            union |= (alpha >= 0.5).astype(np.uint8)
        x, y, box_width, box_height = cv2.boundingRect(union)
        if box_width == 0 or box_height == 0:
            return
        radius = int(math.ceil(JECATATU_SPLIT_DISTANCE * height * ROLE_ALPHA_SCALE / 2.0)) + 1
        # Fecha so na regiao do grupo (com margem do raio): bem mais barato que o quadro.
        rx0, ry0 = max(0, x - radius), max(0, y - radius)
        rx1 = min(union.shape[1], x + box_width + radius)
        ry1 = min(union.shape[0], y + box_height + radius)
        # Retangular: separavel, ~10x mais rapido que a elipse; o poligono apaga a diferenca.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
        fused = np.zeros_like(union, dtype=np.float32)
        fused[ry0:ry1, rx0:rx1] = cv2.morphologyEx(
            union[ry0:ry1, rx0:rx1], cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        key = frozenset(player.slot for player in group)
        shape = self._group_shapes.get(key)
        if shape is None:
            shape = self._group_shapes[key] = SilhouetteShape()
        shape.alpha = fused
        region = self._silhouette_region(shape, width, height)
        if region is None:
            return
        x0, y0, x1, y1, cover = region
        painting = self._slide(self._group_paintings[key], width, height)[y0:y1, x0:x1]
        if cover.dtype == bool:
            cv2.copyTo(painting, cover.view(np.uint8), body[y0:y1, x0:x1])
        else:
            body[y0:y1, x0:x1] = cv2.blendLinear(body[y0:y1, x0:x1], painting, 1.0 - cover, cover)

    def _process_jeca(
        self, frame_bgr: np.ndarray, body_mask: Optional[np.ndarray] = None
    ) -> VisionResult:
        """Jeca: pinturas em sequencia ao fundo e, por cima, as silhuetas recortadas
        em xadrez, um de cada paleta, com os vaos transparentes."""
        height, width = frame_bgr.shape[:2]
        boxes, masks = self._segmenter.people(frame_bgr)
        self._people_source = "YOLO-seg"
        self._people_history.append(min(len(boxes), CAIPIRA_MAX_PEOPLE))
        self.people_count = sorted(self._people_history)[len(self._people_history) // 2]
        self._people_yolo_boxes = boxes
        # Sem arpejos: so o rastreio (sem fluxo optico, que aqui nao seria usado).
        scale = np.ones(2, dtype=np.float32)
        now = self._track_players(
            boxes, None, scale, (width, height), list(range(1, CAIPIRA_MAX_PEOPLE + 1)), False
        )
        self._update_silhouette_masks(boxes, masks, now, width, height)
        state = self._slide_state(now)
        self._prefetch_slide(state[1], width, height, False)
        body = self._compose_slides(state, width, height)
        for player in self._viola_players:
            self._paste_silhouette(body, player)
        debug = frame_bgr.copy()
        self._draw_people_debug(debug)
        return VisionResult(None, debug, body)

    @staticmethod
    def _raster_polygons(polygons: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
        """Poligonos preenchidos (255) numa mascara do tamanho dado."""
        area = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(area, [np.round(polygon).astype(np.int32) for polygon in polygons], 255)
        return area

    def _process_viola(
        self, frame_bgr: np.ndarray, gray: np.ndarray, body_mask: Optional[np.ndarray] = None
    ) -> VisionResult:
        """Viola: pinturas em sequencia; cada pessoa liga um arpejo, mais alto se mexe."""
        height, width = frame_bgr.shape[:2]
        flow = None if self._previous_gray is None else self._farneback(gray)
        scale = np.array((gray.shape[1] / width, gray.shape[0] / height), dtype=np.float32)
        small = self._silhouette_mask(frame_bgr, body_mask)
        self._people_source = "MOG2" if body_mask is None else "Kinect"
        self._count_people(small)
        if self._yolo is not None:
            self._count_people_yolo(frame_bgr)
        boxes = self._people_yolo_boxes if self._yolo is not None else self._people_boxes
        self._mix_arpeggios(boxes, flow, scale, (width, height))
        state = self._slide_state(self._viola_now)
        self._prefetch_slide(state[1], width, height, self.viola_blend == "cores")
        if self.viola_blend == "nenhuma":
            body = self._compose_slides(state, width, height)
        else:
            body = self._blend_plaid(state, flow, scale, width, height)
        debug = frame_bgr.copy()
        self._draw_people_debug(debug)
        return VisionResult(None, debug, body)

    def _draw_people_debug(self, debug: np.ndarray) -> None:
        """Caixas e contagens da mascara e do YOLO, para comparar os dois."""
        # Cada pessoa: o seu canal e o volume, com a excitacao das cordas (caipira)
        # ou o movimento (viola, roda); no caipira a caixa engrossa enquanto ela
        # excita as cordas. Texto proporcional a largura do quadro.
        font_scale = 0.8 * min(1.0, debug.shape[1] / 1280.0) + 0.2
        for player in self._viola_players:
            x0, y0, x1, y1 = player.box
            volume = self.channel_volumes.get(player.slot, 0.0)
            if self.mode == "caipira":
                if player.excitation >= CAIPIRA_EXCITE_MIN:
                    cv2.rectangle(
                        debug, (x0 - 8, y0 - 8), (x1 + 8, y1 + 8), CAIPIRA_PLAYER_COLOR, 6
                    )
                text = f"canal{player.slot} vol {volume:.0%} cordas {player.excitation:.0%}"
            elif self.mode in ("viola", ROLE_MODE):
                text = f"canal{player.slot} vol {volume:.0%} mov {player.motion:.1f}"
            else:
                text = f"pessoa{player.slot}"
            for thickness, ink in ((5, (0, 0, 0)), (2, CAIPIRA_PLAYER_COLOR)):
                cv2.putText(
                    debug, text, (x0, max(25, y0 - 14)), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    ink, thickness, cv2.LINE_AA,
                )
        for x0, y0, x1, y1 in self._people_boxes:
            cv2.rectangle(debug, (x0, y0), (x1, y1), CAIPIRA_MASK_COLOR, 2)
        # Um pouco maiores, para nao sumirem sob as da mascara.
        for x0, y0, x1, y1 in self._people_yolo_boxes:
            cv2.rectangle(debug, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), CAIPIRA_YOLO_COLOR, 2)
        lines = [(f"{self._people_source}: {self.people_count}", CAIPIRA_MASK_COLOR)]
        if self.people_count_yolo is not None:
            lines.append((f"YOLO: {self.people_count_yolo}", CAIPIRA_YOLO_COLOR))
        y = debug.shape[0] - 20 - 45 * (len(lines) - 1)
        for text, color in lines:
            for thickness, ink in ((6, (0, 0, 0)), (2, color)):
                cv2.putText(
                    debug, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, ink, thickness,
                    cv2.LINE_AA,
                )
            y += 45

    def _silhouette_fill(
        self,
        frame_bgr: np.ndarray,
        body_mask: Optional[np.ndarray],
        small: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Quadro preto com os corpos preenchidos pela cor base do xadrez."""
        height, width = frame_bgr.shape[:2]
        if small is None:
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

    def _caipira_background(self, width: int, height: int) -> np.ndarray:
        """Pintura atual ajustada ao quadro e escurecida; refeita so ao trocar."""
        key = (self.landscape_index, width, height)
        if self._background_key != key:
            image = self._landscape_images[self.landscape_index][1]
            fitted = fit_cover(image, width, height, self._crop_anchor(image))
            self._background = cv2.convertScaleAbs(
                fitted, alpha=CAIPIRA_BACKGROUND_BRIGHTNESS
            )
            self._background_key = key
        return self._background

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
        elif self.mode in PLAID_MODES and self._field.layout != (
            self.plaid_layout,
            self.plaid_density,
            self.plaid_rotation,
        ):
            self._field = self._make_field(width, height)
        if self.mode in PLAID_MODES:
            self._field.gain = self.plaid_gain
            self._field.hue = self.plaid_hue
            self._field.disorder_pulses = self.mode == "caipira"
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
        self._field.update(flow, scale, self.FLOW_THRESHOLD)
        if self.mode in (*PLAID_MODES, "paisagens"):
            now = self.clock()
            # Limita o passo: uma pausa longa nao deve teletransportar os pulsos nem
            # adiantar a regeneracao.
            dt = 0.0 if self._field_clock is None else min(now - self._field_clock, 0.1)
            self._field_clock = now
            if self.mode in PLAID_MODES:
                self._field.advance(dt)
                self.line_disorder = self._field.disorder
            else:
                self._field.regen_delay = self.landscape_regen_delay
                self._field.regen_time = self.landscape_regen_time
                self._field.regenerate(dt)
        # O caipira conta as pessoas em todo quadro, com ou sem silhueta; a mesma
        # mascara serve a silhueta, para o MOG2 aprender uma vez so por quadro.
        small = None
        if self.mode == "caipira":
            small = self._silhouette_mask(frame_bgr, body_mask)
            self._people_source = "MOG2" if body_mask is None else "Kinect"
            self._count_people(small)
            if self._yolo is not None:
                self._count_people_yolo(frame_bgr)
            boxes = self._people_yolo_boxes if self._yolo is not None else self._people_boxes
            self._play_toada(boxes, flow, scale, (width, height), self._field)
        # A silhueta fica por baixo; linhas e pulsos sao desenhados por cima. No
        # caipira, a pintura fica por baixo de tudo.
        if self.mode in PLAID_MODES and self.silhouette:
            body = self._silhouette_fill(frame_bgr, body_mask, small)
            if self.mode == "caipira":
                body = cv2.max(self._caipira_background(width, height), body)
        elif self.mode == "caipira":
            body = self._caipira_background(width, height).copy()
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
        if self.mode == "caipira" and self.render.debug_output:
            self._draw_people_debug(debug)
        return VisionResult(None, debug, body)

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

    def _yolo_keypoints(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Keypoints (pessoas, 17, 2) no quadro original e confiancas (pessoas, 17)."""
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
        points, confidence = self._yolo.keypoints(frame_small)
        points = np.asarray(points, dtype=np.float32) * np.array(
            (width / input_width, height / input_height), dtype=np.float32
        )
        return points, confidence

    def _process_yolo(self, frame_bgr: np.ndarray) -> VisionResult:
        height, width = frame_bgr.shape[:2]
        points, confidence = self._yolo_keypoints(frame_bgr)
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


def mp3_gapless(data: bytes) -> Optional[tuple[int, int]]:
    """(amostras a pular, amostras validas) do cabecalho Xing/Info + LAME, ou None."""
    position = 0
    if data[:3] == b"ID3":
        position = 10 + sum((data[6 + index] & 0x7F) << (7 * (3 - index)) for index in range(4))
    while position + 4 <= len(data) and not (
        data[position] == 0xFF and data[position + 1] & 0xE0 == 0xE0
    ):
        position += 1
    if position + 4 > len(data):
        return None
    # MPEG-1 tem 1152 amostras por quadro; MPEG-2/2.5, 576.
    samples_per_frame = 1152 if (data[position + 1] >> 3) & 3 == 3 else 576
    header = data[position : position + 200]
    tag = max(header.find(b"Xing"), header.find(b"Info"))
    if tag < 0:
        return None
    tag += position
    flags = int.from_bytes(data[tag + 4 : tag + 8], "big")
    if not flags & 1:
        return None
    frames = int.from_bytes(data[tag + 8 : tag + 12], "big")
    offset = tag + 12 + 4 * bool(flags & 2) + 100 * bool(flags & 4) + 4 * bool(flags & 8)
    lame = data[offset : offset + 24]
    if len(lame) < 24 or not lame[:4].isalpha():
        return None
    delay = (lame[21] << 4) | (lame[22] >> 4)
    padding = ((lame[22] & 0x0F) << 8) | lame[23]
    return delay + MP3_DECODER_DELAY, frames * samples_per_frame - delay - padding


def load_viola_channels(directory: str, sample_rate: int) -> dict[int, np.ndarray]:
    """Arpejo de cada canal (1 a VIOLA_CHANNELS), mono, pelo "canalN" no nome."""
    import miniaudio

    channels: dict[int, np.ndarray] = {}
    for name in sorted(os.listdir(directory)):
        match = re.search(r"canal(\d+)", name.lower())
        if match is None or not name.lower().endswith(VIOLA_EXTENSIONS):
            continue
        channel = int(match.group(1))
        if not 1 <= channel <= VIOLA_CHANNELS:
            continue
        if channel in channels:
            print(f"Aviso: canal{channel} repetido; ignorando {name}.", file=sys.stderr)
            continue
        with open(os.path.join(directory, name), "rb") as sound_file:
            data = sound_file.read()
        sound = miniaudio.decode(
            data,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1,
            sample_rate=sample_rate,
        )
        samples = np.asarray(sound.samples, dtype=np.float32)
        gapless = mp3_gapless(data) if name.lower().endswith(".mp3") else None
        if gapless is not None:
            skip, length = gapless
            # Converte da taxa do arquivo para a de saida.
            ratio = sample_rate / sound_file_rate(data, sound)
            # O miniaudio recente (dr_mp3) ja corta atraso e enchimento ao ler o
            # cabecalho LAME e devolve so as amostras validas: cortar de novo comeria
            # o inicio do loop.
            if abs(len(samples) - length * ratio) > 2 + ratio:
                samples = samples[round(skip * ratio) : round((skip + length) * ratio)]
        elif name.lower().endswith(".mp3"):
            print(
                f"Aviso: {name} sem cabecalho LAME/Info; o loop pode ter um pequeno "
                "silencio. Exporte com o cabecalho (padrao do Audacity/ffmpeg) ou use WAV.",
                file=sys.stderr,
            )
        peak = float(np.abs(samples).max(initial=0.0))
        if peak > 0:
            samples *= VIOLA_CHANNEL_PEAK / peak
        channels[channel] = samples
    lengths = {round(len(loop) / sample_rate, 2) for loop in channels.values()}
    if len(lengths) > 1:
        print(
            f"Aviso: arpejos com duracoes diferentes ({', '.join(f'{v:g}s' for v in sorted(lengths))}); "
            "cada um repete no seu tempo e eles saem de sincronia.",
            file=sys.stderr,
        )
    return channels


def sound_file_rate(data: bytes, sound) -> int:
    """Taxa original do arquivo (o decode ja devolve convertida)."""
    import miniaudio

    try:
        return miniaudio.mp3_get_info(data).sample_rate
    except Exception:
        return sound.sample_rate


class ViolaArpeggios:
    """Arpejos em loop, sincronizados; cada canal tem um volume com rampa.

    set_volume() pode ser chamado de qualquer thread; vale no proximo bloco.
    """

    def __init__(
        self, directory: str = VIOLA_ARPEGGIO_DIR, sample_rate: int = VIOLA_SAMPLE_RATE
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = load_viola_channels(directory, sample_rate)
        if not self.channels:
            raise RuntimeError(
                f"nenhum arquivo com canal1 a canal{VIOLA_CHANNELS} no nome em {directory}"
            )
        self.error: Optional[str] = None
        self._targets = dict.fromkeys(self.channels, 0.0)
        self.volumes = dict.fromkeys(self.channels, 0.0)
        # Relogio unico: todos os canais leem a mesma posicao, mesmo mudos, mais o
        # deslocamento (amostras) de cada um: 0 no modo viola (sincronizados).
        self._position = 0
        self._offsets = dict.fromkeys(self.channels, 0)
        # Tempo proprio de cada canal (jecatatu): um deslocamento sorteado no loop.
        rng = np.random.default_rng()
        self.own_offsets = {
            channel: int(rng.integers(len(loop))) for channel, loop in self.channels.items()
        }
        # Troca de tempo em andamento: canal -> (deslocamento novo, progresso 0 a 1).
        self._fades: dict[int, tuple[int, float]] = {}
        self._lock = threading.Lock()
        self._stream = None
        try:
            import sounddevice

            self._stream = sounddevice.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
                latency=VIOLA_LATENCY,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            self.error = f"{type(exc).__name__}: {exc}"
            print(
                f"Aviso: arpejos sem audio ({self.error}); os canais aparecem no debug, "
                "mas nao tocam.",
                file=sys.stderr,
            )

    def set_volume(self, channel: int, volume: float) -> None:
        if channel in self._targets:
            self._targets[channel] = volume

    def follow(self, channel: int, leader: int) -> None:
        """Poe o canal no tempo proprio de leader (ele mesmo: volta ao seu), com
        crossfade de VIOLA_SYNC_FADE s; o mesmo tempo de antes nao faz nada."""
        if channel not in self._offsets or leader not in self.own_offsets:
            return
        target = self.own_offsets[leader]
        with self._lock:
            fade = self._fades.get(channel)
            heading = fade[0] if fade is not None else self._offsets[channel]
            if heading == target:
                return
            if self.volumes[channel] == 0.0 and self._targets[channel] == 0.0:
                # Mudo (a pessoa acabou de chegar): vai direto, nao ha o que ouvir.
                self._offsets[channel] = target
                self._fades.pop(channel, None)
                return
            if fade is not None:
                # Troca no meio de outra: parte do tempo para onde ja ia.
                self._offsets[channel] = fade[0]
            self._fades[channel] = (target, 0.0)

    def leader_of(self, channel: int) -> int:
        """Canal cujo tempo proprio o canal segue (ou para onde esta indo)."""
        fade = self._fades.get(channel)
        offset = fade[0] if fade is not None else self._offsets.get(channel)
        for leader, own in self.own_offsets.items():
            if own == offset:
                return leader
        return channel

    def render(self, frames: int) -> np.ndarray:
        """Proximas frames amostras da mistura (mono, float32)."""
        out = np.zeros(frames, dtype=np.float32)
        indices = np.arange(self._position, self._position + frames)
        rise = VIOLA_VOLUME_RISE * frames / self.sample_rate
        fall = VIOLA_VOLUME_FALL * frames / self.sample_rate
        step = frames / (VIOLA_SYNC_FADE * self.sample_rate)
        with self._lock:
            for channel, loop in self.channels.items():
                current = self.volumes[channel]
                target = self._targets[channel]
                following = current + min(rise, max(-fall, target - current))
                self.volumes[channel] = following
                fade = self._fades.get(channel)
                if current == 0.0 and following == 0.0:
                    # Mudo: a troca de tempo termina na hora (nada para ouvir).
                    if fade is not None:
                        self._offsets[channel] = fade[0]
                        del self._fades[channel]
                    continue
                ramp = np.linspace(current, following, frames, endpoint=False, dtype=np.float32)
                sound = loop[(indices + self._offsets[channel]) % len(loop)]
                if fade is not None:
                    # Duas cabecas de leitura no mesmo loop: a antiga sai, a nova entra.
                    target_offset, progress = fade
                    mix = np.linspace(
                        progress, min(1.0, progress + step), frames, endpoint=False,
                        dtype=np.float32,
                    )
                    sound = sound * (1.0 - mix) + loop[(indices + target_offset) % len(loop)] * mix
                    if progress + step >= 1.0:
                        self._offsets[channel] = target_offset
                        del self._fades[channel]
                    else:
                        self._fades[channel] = (target_offset, progress + step)
                out += ramp * sound
        self._position += frames
        return np.tanh(out, out=out)

    def _callback(self, outdata, frames, time_info, status) -> None:
        outdata[:, 0] = self.render(frames)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def make_viola_arpeggios(mode: str, toada: bool = True) -> Optional[ViolaArpeggios]:
    """Canais em loop do modo: a toada do caipira (toada=False a desliga) ou os
    arpejos da viola, roda e jecatatu; None em outro modo ou sem sons."""
    if mode == "caipira":
        if not toada:
            return None
        directory, label = CAIPIRA_CHANNEL_DIR, "toada"
    elif mode in ("viola", ROLE_MODE, JECATATU_MODE):
        directory, label = VIOLA_ARPEGGIO_DIR, "arpejos"
    else:
        return None
    try:
        return ViolaArpeggios(directory)
    except Exception as exc:
        print(f"Aviso: {label} desativados: {exc}", file=sys.stderr)
        return None


class SilhouetteShape:
    """Forma de silhueta sem pessoa propria (a fusao de um grupo): mascara e poligono
    estabilizado, como os de ViolaPlayer."""

    def __init__(self) -> None:
        self.alpha: Optional[np.ndarray] = None
        self.polygons: Optional[list[np.ndarray]] = None
        self.polygon_center = np.zeros(2, dtype=np.float32)


class ViolaPlayer:
    """Pessoa rastreada no caipira, na viola ou na roda, com o canal que ela toca."""

    def __init__(self, box: tuple[int, int, int, int], slot: int, now: float) -> None:
        # Canal da toada (caipira) ou de arpejo (viola, roda).
        self.slot = slot
        self.seen = now
        self.motion = 0.0
        # Caipira: excitacao das cordas na caixa (0 a 1) no ultimo quadro.
        self.excitation = 0.0
        # Viola: movimento suavizado (0 a 1) que da o volume do canal.
        self.level = 0.0
        # Roda de viola: paisagem da pessoa e a sua mascara suavizada (meia resolucao).
        self.painting: Optional[int] = None
        self.alpha: Optional[np.ndarray] = None
        # Roda de viola: poligonos atuais da silhueta (escala da mascara pequena).
        self.polygons: Optional[list[np.ndarray]] = None
        # Centro da mascara a que os poligonos atuais estao presos.
        self.polygon_center = np.zeros(2, dtype=np.float32)
        # Jecatatu: brilho da silhueta (1 no contato, decai ate 0).
        self.glow = 0.0
        # Jeca: tecido de xadrez da pessoa (cores, linhas, vetor de repeticao, margem,
        # tamanho do quadro e angulo), centro em que ele nasceu e deslocamento atual.
        self.plaid: Optional[tuple] = None
        self.plaid_origin = np.zeros(2, dtype=np.float32)
        self.plaid_offset = np.zeros(2, dtype=np.float32)
        self.place(box, now)

    def place(self, box: tuple[int, int, int, int], now: float) -> None:
        self.box = box
        self.center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        self.seen = now


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
        choices=(
            "kinect", "mog2", "optical_flow", "conexoes", "capim", "mato", "paisagens",
            *PLAID_MODES, "viola", ROLE_MODE, JECA_MODE, JECATATU_MODE,
        ),
        default=None,
        help="algoritmo 2D: kinect, mog2, optical_flow, conexoes (malha YOLO), capim "
        "(capim movido pelo optical flow), mato (tracos em camadas, o do meio movido pelo "
        "optical flow), paisagens (imagem borrada pelo optical flow), xadrez (xadrez "
        "Madras movido pelo optical flow), caipira (xadrez com pinturas e viola), viola "
        "(pinturas em sequencia e arpejos ligados pelas pessoas), roda-de-viola "
        "(silhuetas recortadas em paisagens sobre branco, com arpejos), jeca "
        "(pinturas em sequencia com as silhuetas em xadrez transparente) ou jecatatu "
        "(silhuetas em paisagens sobre preto, xadrez reagindo a cabeca, maos e pes, "
        "arpejos que sincronizam quando as silhuetas se fundem) "
        f"(padrao: {DEFAULT_MODE})",
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
        "--rotation",
        type=float,
        default=None,
        help="xadrez: rotacao do padrao em graus, em torno do centro do quadro "
        f"(padrao: {DEFAULT_PLAID_ROTATION:g}; {CAIPIRA_ROTATION:g} no caipira; teclas d/c)",
    )
    parser.add_argument(
        "--hue",
        type=float,
        default=None,
        help="xadrez: matiz base das linhas em graus (0 vermelho, 120 verde, 240 azul); "
        "as faixas variam so em luminosidade, da cor pura ao branco "
        f"(padrao: sorteado; {CAIPIRA_HUE:g}, laranja, no caipira; teclas f/v giram o "
        "disco de cores)",
    )
    parser.add_argument(
        "--pulses",
        type=float,
        default=DEFAULT_PULSE_RATE,
        help="xadrez: pulsos de neon que nascem por segundo, em media, em intervalos "
        "aleatorios; 0 desliga (padrao: %(default)s)",
    )
    parser.add_argument(
        "--silhouette",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="silhueta dos corpos (mascara de profundidade no Kinect, MOG2 nas outras "
        "fontes; tecla g liga/desliga): no xadrez e preenchida pela cor base; no "
        "paisagens e uma janela para a proxima pintura da lista",
    )
    parser.add_argument(
        "--layout",
        choices=tuple(PLAID_LAYOUTS),
        default=None,
        help="xadrez: mix (malhas independentes em alturas diferentes, com densidades, "
        f"angulos e paralaxe proprios; densidade ate {PLAID_MIX_MAX_DENSITY}) ou regular "
        "(uma malha) (padrao: mix; regular no modo viola, onde o mix dobra o custo; "
        "tecla l alterna)",
    )
    parser.add_argument(
        "--blend",
        choices=VIOLA_BLENDS,
        default=VIOLA_BLENDS[0],
        help="viola: mescla do xadrez com as pinturas; tecido (a pintura se deforma "
        "com as linhas), quadros (cada quadrado segue com atraso proprio e troca de "
        "pintura no seu momento), cores (linhas com as cores da pintura), janela (a "
        "pintura so aparece pelas linhas), recorte (as linhas recortam a pintura e so "
        "aparecem quando se movem) ou nenhuma "
        "(padrao: %(default)s; tecla e alterna)",
    )
    parser.add_argument(
        "--pi",
        action="store_true",
        help="perfil Raspberry Pi 4: modo caipira (padrao), xadrez ou paisagens; so a janela do "
        "projetor (FPS no terminal); no xadrez e no caipira, sem anti-aliasing, curvas e silhueta "
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
        "a/z alternam entre elas em ordem alfabetica; caipira: fundo das linhas, uma "
        "sorteada na partida, tecla i passa a proxima (padrao: a pasta paisagens/)",
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
        "--viola",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="caipira: cada pessoa detectada (ate 5) recebe um canal sorteado da toada, "
        "que sobe de volume enquanto ela excita as cordas e decai sem interacao; sons "
        "em sons/toada-caipira (padrao: ligado)",
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
            parser.error(
                f"--pi so suporta --mode {', '.join(PI_MODES[:-1])} ou {PI_MODES[-1]}"
            )
    if args.mode is None:
        args.mode = DEFAULT_MODE
    if args.layout is None:
        args.layout = "regular" if args.mode in ("viola", JECATATU_MODE) else "mix"
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
    plaid_rotation: Optional[float] = None,
    plaid_hue: Optional[float] = None,
    pulse_rate: float = DEFAULT_PULSE_RATE,
    silhouette: bool = False,
    plaid_layout: str = "mix",
    render: PlaidRender = PlaidRender(),
    people_yolo: bool = False,
    arpeggios: Optional[ViolaArpeggios] = None,
    viola_blend: str = VIOLA_BLENDS[0],
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
            plaid_rotation=plaid_rotation,
            plaid_hue=plaid_hue,
            pulse_rate=pulse_rate,
            silhouette=silhouette,
            plaid_layout=plaid_layout,
            render=render,
            people_yolo=people_yolo,
            arpeggios=arpeggios,
            viola_blend=viola_blend,
        )
    )


# Teclas por modo, impressas na partida; q, m e as do metronomo valem sempre.
COMMON_KEY_HELP = "q sai | m troca a fonte | t metronomo, - / = andamento"
MODE_KEY_HELP = {
    "kinect": "a/z profundidade minima | s/x maxima | b calibra o fundo (sala vazia)",
    "xadrez": "a/z densidade | s/x ganho | d/c rotacao | f/v matiz | g silhueta | "
    "l malha mix/regular | b fundo (Kinect)",
    "caipira": "a/z densidade | s/x ganho | d/c rotacao | f/v matiz | g silhueta | "
    "l malha mix/regular | i proxima pintura | b fundo (Kinect)",
    ROLE_MODE: "e borda poligono/serrilhado/suave",
    JECA_MODE: "i proxima pintura | e borda poligono/serrilhado/suave",
    JECATATU_MODE: "e borda poligono/serrilhado/suave | a/z densidade | s/x ganho | "
    "d/c rotacao | f/v matiz | l malha mix/regular",
    "viola": "i proxima pintura | e mescla tecido/quadros/cores/janela/recorte/nenhuma | "
    "a/z densidade | "
    "s/x ganho | d/c rotacao | f/v matiz | l malha mix/regular | b fundo (Kinect)",
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
    arpeggios = make_viola_arpeggios(active_mode, args.viola)
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
            plaid_rotation=args.rotation,
            plaid_hue=args.hue,
            pulse_rate=args.pulses,
            silhouette=args.silhouette,
            plaid_layout=args.layout,
            render=render,
            people_yolo=not args.pi,
            arpeggios=arpeggios,
            viola_blend=args.blend,
        )
    except Exception as exc:
        capturer.close()
        if arpeggios is not None:
            arpeggios.close()
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
                            plaid_rotation=args.rotation,
                            plaid_hue=args.hue,
                            pulse_rate=args.pulses,
                            silhouette=args.silhouette,
                            plaid_layout=args.layout,
                            render=render,
                            people_yolo=not args.pi,
                            arpeggios=arpeggios,
                            viola_blend=args.blend,
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
            elif active_mode in PLAID_MODES:
                if active_mode == "caipira" and key == CAIPIRA_IMAGE_KEY:
                    processor.switch_landscape(1)
                    print(processor.label)
                elif key in PLAID_KEY_ADJUSTMENTS:
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
            elif active_mode in ("viola", JECA_MODE) and key == CAIPIRA_IMAGE_KEY:
                processor.skip_slide()
                print(processor.label)
            elif active_mode in (ROLE_MODE, JECA_MODE, JECATATU_MODE) and key == ROLE_EDGE_KEY:
                processor.cycle_role_edge()
                print(processor.label)
            elif active_mode == "viola" and key == VIOLA_BLEND_KEY:
                processor.cycle_blend()
                print(processor.label)
            elif active_mode in ("viola", JECATATU_MODE) and key in PLAID_KEY_ADJUSTMENTS:
                processor.adjust_plaid(**PLAID_KEY_ADJUSTMENTS[key])
                print(processor.label)
            elif active_mode in ("viola", JECATATU_MODE) and key == LAYOUT_TOGGLE_KEY:
                processor.toggle_layout()
                print(processor.label)
            elif active_mode == "viola" and key == ord("b") and isinstance(
                capturer, KinectV1Capturer
            ):
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
        if arpeggios is not None:
            arpeggios.close()
        if worker is not None:
            worker.close()
        if capturer is not None:
            capturer.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# Dança Interativa: PoC

Prova de conceito em Python para explorar visuais generativos dirigidos pelo movimento, usando Kinect v1 (1414/1473), webcam USB ou arquivo MP4/MOV.

O modo padrão é o **xadrez**, documentado à parte em [docs/xadrez.md](docs/xadrez.md): como rodar, teclas, como funciona, desempenho medido e limitações. O modo **paisagens** — pinturas que os corpos borram e espalham — tem a sua documentação em [docs/paisagens.md](docs/paisagens.md). Um PNG do padrão com fundo transparente está em [docs/img/xadrez_transparente.png](docs/img/xadrez_transparente.png).

## Primeiros passos

Precisa de Linux (testado no Ubuntu e pensado para o Raspberry Pi OS Bookworm), **Python 3.10 ou mais novo** e uma sessão gráfica: as janelas são do OpenCV e não abrem por SSH sem tela (por SSH, com a sessão gráfica aberta e logada na máquina, use `export DISPLAY=:0`). A PoC confere a versão do Python, os pacotes e a tela na partida e explica o que falta.

### Notebook ou desktop com webcam

```bash
git clone https://github.com/rodjuncode/somos-teia.git
cd somos-teia
sudo apt-get install python3-venv libportaudio2   # libportaudio2: som do metrônomo (opcional)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pi.txt       # leve: todos os modos menos conexoes (pulsos do xadrez sorteados)
python dance_interactive_poc.py --source webcam   # abre no modo xadrez (o padrão)
```

Para usar também o modo `conexoes` e os pulsos do xadrez nascendo das mãos (ambos usam YOLO), instale a versão completa. Sem placa NVIDIA, instale antes o PyTorch só de CPU — o padrão do pip traz as bibliotecas CUDA, alguns GB a mais:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # só sem NVIDIA
python -m pip install -r requirements.txt
python dance_interactive_poc.py --source webcam --mode conexoes
```

No primeiro uso do `conexoes` a PoC baixa `yolov8n-pose.pt` (~7 MB) e, em CPU, exporta o modelo para OpenVINO INT8, o que leva alguns minutos e baixa um pequeno dataset de calibração; tudo isso precisa de internet e fica guardado ao lado do script para as próximas execuções.

### Raspberry Pi 4 com Kinect

Use o Raspberry Pi OS Bookworm 64 bits (Python 3.11), com dissipador e ventoinha, e o Kinect na sua própria fonte de energia.

```bash
git clone https://github.com/rodjuncode/somos-teia.git
cd somos-teia
sudo apt-get install python3-venv python3-dev build-essential libfreenect-dev libfreenect-bin libportaudio2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pi.txt -r requirements-kinect.txt
freenect-glview                                     # opcional: confirma que o Kinect responde
python dance_interactive_poc.py --pi                # xadrez; ou --pi --mode paisagens
```

O primeiro quadro do Kinect leva ~7 s. Sem Kinect, a PoC cai para a webcam sozinha. Os vídeos de ensaio não estão no repositório (pasta `videos/` ignorada); sem câmera, use qualquer MP4/MOV com `--source arquivo.mp4`.

Na partida, a PoC imprime as teclas do modo ativo; elas valem com a janela da PoC em foco. Detalhes de cada modo, desempenho e opções estão nas seções abaixo.

## Estado atual

O protótipo em [`dance_interactive_poc.py`](dance_interactive_poc.py) já implementa:

- Captura contínua em thread separada, mantendo o frame mais recente disponível.
- Processamento visual (YOLO em CUDA, MOG2, Farneback ou máscara Kinect) em outra thread: enquanto um frame é exibido, o seguinte já está sendo processado, e a thread principal só desenha o overlay, atualiza as janelas e lê o teclado. Com inferência em CPU (OpenVINO ou PyTorch CPU), o processamento continua na thread principal: com a CPU saturada, sobrepor a pintura das janelas deixava cada inferência ~6 ms mais lenta sem aumentar o FPS.
- Três fontes intercambiáveis pela CLI e pela tecla `m`: Kinect v1, webcam USB e arquivo MP4/MOV em loop.
- Fallback automático de Kinect para webcam e, opcionalmente, arquivo de vídeo.
- Saída padronizada de todos os capturadores: sucesso, frame BGR (formato nativo do OpenCV), profundidade e máscara corporal; webcam e vídeo devolvem `None` nesses dois últimos.
- Webcam e vídeo usam modos selecionáveis: MOG2, fluxo óptico Farneback, `conexoes` (malha entre dançarinos com YOLOv8n-pose), `capim` (capim movido pelo fluxo óptico), `mato` (traços em camadas mudados pelo fluxo), `paisagens` (pinturas borradas pelo fluxo) ou `xadrez` (xadrez Madras movido pelo fluxo óptico). O padrão é o `xadrez`, laranja e com a silhueta dos corpos ligada, sem inferência de IA.
- Modo Kinect mantém profundidade RAW; se a inicialização falhar, o fallback usa MOG2 na webcam/vídeo.
- Máscara binária por faixa de profundidade no Kinect, configurada inicialmente entre 800 e 3000 mm (ajustável por linha de comando e por atalhos).
- Overlay de máscara/contornos, vetores ópticos ou esqueleto YOLO, FPS, modo ativo e latência estimada.
- Duas janelas OpenCV neste momento: debug e projetor Corpo/Frontal. As telas de Chão/Fundo e Simulação/Consolidada e todo o cálculo associado estão temporariamente removidos.
- Encerramento de captura e janelas ao pressionar `q`.

Os modos YOLO (CUDA e OpenVINO), MOG2, Farneback e Kinect foram executados de ponta a ponta com os vídeos de ensaio e o Kinect v1 conectado; os backends YOLO diretos produzem keypoints idênticos aos do Ultralytics. A exibição nos projetores reais ainda não foi validada. As metas de latência são objetivos por algoritmo, não garantias ponta a ponta.

## Requisitos

- Linux/Ubuntu, Python 3.10 ou mais novo e `python3-venv`.
- Para Kinect v1: dispositivo conectado, bibliotecas de desenvolvimento `libfreenect` e binding Python `freenect`.
- Para webcam/fallback: câmera acessível e dependências Python do [`requirements.txt`](requirements.txt).

O Kinect requer a biblioteca nativa `libfreenect` e headers de desenvolvimento; o metrônomo usa `sounddevice`, que precisa da `libportaudio2` do sistema (sem ela a PoC roda normalmente, só sem som); a binding Python e as demais dependências ficam isoladas no venv. O pacote `python3-freenect` não é usado. Ultralytics/PyTorch adiciona dependências grandes e baixa `yolov8n-pose.pt` ao primeiro uso do modo YOLO.

```bash
sudo apt-get update
sudo apt-get install libfreenect-dev libfreenect-bin libportaudio2 python3-dev python3-venv build-essential
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-kinect.txt
```

O comando pip instala as dependências dentro do venv. São três listas: [`requirements.txt`](requirements.txt) (completa, com YOLO), [`requirements-pi.txt`](requirements-pi.txt) (leve, sem YOLO: todos os modos menos `conexoes`) e [`requirements-kinect.txt`](requirements-kinect.txt) (só a binding `freenect`, para somar a uma das outras; o pip instala sozinho o Cython e o NumPy de que ela precisa para compilar). Para YOLO, use `--mode conexoes`; MOG2 e Farneback não precisam de rede neural.

Para conferir a binding antes de conectar a câmera, execute `python -c "import freenect; print('freenect importado')"` dentro do venv. A disponibilidade do dispositivo e as permissões USB ainda precisam ser verificadas no hardware usado.

## Executar

Com o ambiente ativado, a fonte padrão é Kinect, com fallback automático para webcam:

```bash
python dance_interactive_poc.py
```

Escolha explicitamente a fonte:

```bash
python dance_interactive_poc.py --source kinect
python dance_interactive_poc.py --source webcam
python dance_interactive_poc.py --source ensaio.mp4
python dance_interactive_poc.py --source kinect --fallback-video ensaio.mp4
```

Selecione o algoritmo de processamento para webcam ou vídeo:

```bash
python dance_interactive_poc.py --source ensaio.mp4 --mode mog2
python dance_interactive_poc.py --source ensaio.mp4 --mode optical_flow
python dance_interactive_poc.py --source ensaio.mp4 --mode conexoes
python dance_interactive_poc.py --source ensaio.mp4 --mode conexoes --nogpu
python dance_interactive_poc.py --source ensaio.mp4 --mode conexoes --nogpu --no-int8
python dance_interactive_poc.py --source ensaio.mp4 --mode conexoes --max-distance 150 --max-connections 5 --point-deadband 4
python dance_interactive_poc.py --source ensaio.mp4 --mode conexoes --show-points
python dance_interactive_poc.py --source ensaio.mp4 --mode conexoes --edge-style bezier
python dance_interactive_poc.py --source ensaio.mp4 --mode capim
python dance_interactive_poc.py --source ensaio.mp4 --mode capim --grass-density 40 --grass-height 0.6 --grass-gain 1.5
python dance_interactive_poc.py --source ensaio.mp4 --mode mato
python dance_interactive_poc.py --source ensaio.mp4 --mode mato --density 1.5 --gain 2
python dance_interactive_poc.py --source ensaio.mp4 --mode paisagens
python dance_interactive_poc.py --source ensaio.mp4 --mode paisagens --image minha_paisagem.jpg --image-anchor 0.3 --gain 0.5
python dance_interactive_poc.py --source ensaio.mp4 --mode paisagens --image minhas_pinturas/
python dance_interactive_poc.py --source ensaio.mp4 --mode paisagens --palette cyberpunk
python dance_interactive_poc.py --source ensaio.mp4 --mode xadrez
python dance_interactive_poc.py --source ensaio.mp4 --mode xadrez --density 1.4 --gain 1.5 --rotation 30 --hue 200
python dance_interactive_poc.py --source ensaio.mp4 --mode xadrez --pulses 10
python dance_interactive_poc.py --source ensaio.mp4 --mode xadrez --silhouette
python dance_interactive_poc.py --source ensaio.mp4 --mode xadrez --layout regular
python dance_interactive_poc.py --source ensaio.mp4 --mode xadrez --metronome --bpm 90
python dance_interactive_poc.py --source kinect --mode kinect
```

O modo padrão, para qualquer fonte, é o `xadrez` (laranja, layout `mix`, silhueta ligada; veja [docs/xadrez.md](docs/xadrez.md)). `--mode kinect` exige `--source kinect`; os outros modos podem processar RGB de webcam, vídeo ou Kinect. `m` continua alternando as fontes disponíveis sem reiniciar e reseta o estado temporal do algoritmo quando necessário.

- **MOG2:** `BackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)` roda em 160×120; a máscara é ampliada ao tamanho original, limpa com abertura/fechamento morfológicos e seu centroide vem dos momentos.
- **Optical Flow:** Farneback roda em 80×60 entre frames cinza consecutivos (o fluxo é zerado a mais de 5 células de onde o cinza mudou mais de 2 níveis: o Farneback inventa até ~3 px/quadro de movimento em áreas lisas ou estouradas mesmo entre quadros idênticos, o que borrava e empurrava o desenho com a cena parada; o filtro vale para todos os modos que usam fluxo e custa ~0,02 ms); a máscara seleciona movimento acima de 0,15 px nessa escala (~1,2 px na imagem original) e a janela Corpo mostra vetores ampliados e recortados pela máscara.
- **Conexões (YOLO):** `yolov8n-pose.pt`, inferência em `imgsz=320` com até 16 detecções por frame para limitar o pós-processamento; cada dançarino gera sete nós: cabeça (média de nariz/olhos/orelhas) e um nó por classe bilateral (ombro, cotovelo, pulso/mão, quadril, joelho e tornozelo), usando o ponto esquerdo quando visível e o direito como fallback, sem criar um ponto médio artificial. Os círculos dos nós ficam ocultos por padrão; `--show-points` os habilita. `--point-deadband` (padrão 4 px) ignora tremor pequeno sem EMA; movimentos maiores avançam no mesmo frame. As distâncias entre os nós que ainda têm conexões livres são calculadas de uma vez em NumPy, respeitando `--max-distance` (padrão 150 px), e as arestas mais próximas são escolhidas até `--max-connections` por nó (padrão 5). Só se ligam nós de corpos diferentes: pares do mesmo dançarino são descartados antes da ordenação, então uma pessoa sozinha não forma malha. Com no máximo 112 nós (16 dançarinos × 7), essa busca vetorizada mede 0,1–0,6 ms, de 5× a 80× mais rápida que o quadtree em Python usado antes. Conexões são persistentes: depois de criada, uma aresta não é recalculada — o desenho apenas acompanha os dois nós, mesmo que se afastem além de `--max-distance` — e só é liberada quando um dos nós deixa de ser detectado; a busca considera apenas nós com conexões livres. Cada haste recebe, ao nascer, uma cor sorteada ao acaso e levada ao brilho máximo (o maior canal vale 255), e a mantém enquanto existir; todas têm espessura fixa de 3 px (`MESH_THICKNESS`), independente da distância entre os nós; os segmentos rígidos YOLO são ignorados. `--edge-style bezier` troca as hastes retas por curvas Bezier cúbicas com as mesmas pontas: os dois pontos de controle são sorteados quando a conexão nasce e guardados em coordenadas relativas à aresta (posição ao longo dela e desvio perpendicular de até 50% do comprimento), então a curva mantém a forma enquanto acompanha os nós — esticando, encolhendo e girando com eles — e é descartada junto com a conexão. Cada curva é desenhada com 20 amostras; com 16 pessoas em 720p, o desenho custa ~1,4 ms a mais por frame que as hastes.
- **Capim:** usa o mesmo Farneback do Optical Flow (quadro cinza reduzido a 1/8) para empurrar um capim alto desenhado com Bezier cúbicas finas (1 px, anti-aliased). No primeiro frame são sorteadas `--grass-density` hastes por 100 px de largura (padrão 25, 320 hastes em 1280 px), cada uma com base no sexto mais baixo do quadro, altura entre metade e o total de `--grass-height` (fração da altura do quadro, padrão 0,5) e uma forma de repouso sutil (inclinação e desvio dos pontos de controle). O único estado dinâmico de cada haste é o deslocamento horizontal da ponta, uma mola amortecida: a cada frame o fluxo horizontal é amostrado nos três pontos de controle (ignorando movimento abaixo do limiar do Optical Flow) e empurra a ponta, com força ajustada por `--grass-gain` (padrão 1). Os controles seguem a ponta num perfil ~t² de haste presa na base e descem para conservar o comprimento; depois do empurrão o capim balança e volta ao repouso. Trocar de fonte zera o movimento mas mantém o capim; um quadro de outro tamanho o sorteia de novo. Os argumentos de `conexoes` (`--max-distance`, `--edge-style` etc.) não se aplicam. O capim é desenhado uma vez no quadro do projetor e composto no de debug com `cv2.max` (por isso some sobre áreas muito claras do vídeo de debug). Medido em 1280×720 com vídeo sintético: ~7 ms por frame com 320 hastes e ~10 ms com 640, sem GPU.
- **Mato:** uma paisagem quase impressionista de traços retos e finos (2 px, anti-aliased), cada um com posição, comprimento, ângulo e cor sorteados no início. O comprimento cresce com a proximidade da camada, o que dá profundidade: na base, 0,6% a 3% da altura do quadro; no meio, 3% a 10%; na frente, 6% a 18% (`MEADOW_LAYERS`). As cores vêm de `MEADOW_PALETTE` (cor e peso, fácil de trocar): ~82% verdes, do verde bandeira ao verde água, ~13% marrons e ~4% rosas e roxos. São três camadas, da mais distante à mais próxima: a **base** (35 000 traços em 1280×720, 70% do brilho — mais traços porque são curtos) e a **frente** (600 traços, que tampam ~17% da tela) são estáticas e desenhadas uma única vez em imagens guardadas — por quadro custam uma cópia e uma composição com máscara (~0,5 ms); a **do meio** (2 000 traços, 90% do brilho) vive com os corpos: cada traço é rígido, o fluxo óptico amostrado nas duas pontas arrasta o centro e a diferença entre elas o gira. Não há mola: as mudanças são permanentes, como se os dançarinos mudassem a paisagem — abrem caminhos, empurram traços à frente e os penteiam na direção do movimento. Um atrito para o traço quando o empurrão acaba, e o empurrão é calibrado para que um traço tocado ande na velocidade do corpo (com `--gain 1`). Os traços ficam presos dentro do quadro, então os que são empurrados para fora se acumulam na borda. A paisagem original só volta com um reset (ao trocar de fonte com `m`); numa sessão longa, o meio vai sendo rearranjado sem limite. As quantidades escalam com a área do quadro e com `--density`; `--gain` regula o empurrão. Com as três camadas, ~97% da tela fica coberta. Medido: ~17–18 ms por quadro em 1280×720 (quase tudo é redesenhar a camada do meio) e ~5,5 ms em 640×480 com 1 thread; montar as camadas no primeiro quadro leva ~125 ms. Desenhar o meio fino e engrossar com `dilate` economizaria ~2 ms com aparência quase igual.
- **Paisagens** (documentação completa em [docs/paisagens.md](docs/paisagens.md))**:** uma imagem cobre a tela sem distorcer. `--image` aceita uma pasta (padrão: [`paisagens/`](paisagens/), com pinturas de Almeida Júnior e Aurélio de Figueiredo) ou um arquivo; com uma pasta, todas as imagens são lidas na partida, em ordem alfabética, e `a` (anterior) e `z` (próxima) alternam entre elas — a nova imagem começa limpa, sem as pinceladas da anterior, e a troca custa ~15 ms num quadro. O nome e a posição aparecem no overlay (`Paisagens: almeida-junior-salto-de-itu-1886 (4/6) | forca 1.0`). Cada imagem é ampliada até cobrir o quadro e o excesso é cortado. Nas imagens em retrato (mais altas que largas), a altura do recorte é sorteada toda vez que a imagem é carregada — na partida, a cada `a`/`z` e também para a pintura da janela da silhueta —, então a mesma pintura volta com enquadramentos diferentes; o valor atual aparece no overlay (`recorte 71%`, de 0% = topo a 100% = base). Nas imagens em paisagem, `--image-anchor` escolhe a faixa visível quando o corte é vertical (0 = topo, 0,5 = meio, o padrão; 1 = base). O fluxo óptico dos corpos borra e espalha a tinta: a cada quadro a tela é reamostrada ao longo do fluxo com `cv2.remap` (cada pixel busca a cor de onde o movimento veio), então onde nada se move a imagem fica intacta, pixel a pixel, e onde há movimento a tinta é arrastada. Um ruído fixo e fino — as "cerdas" (`LANDSCAPE_BRISTLE`) — faz pixels vizinhos andarem quantidades diferentes, e o rastro vira estrias ao longo do gesto, como pinceladas; sem ele o resultado era um marmorizado liso. A espessura desses feixes é `--bristle-width` (em px do quadro processado, padrão 2): 1–2 px dão estrias finas, ~6 px pinceladas largas e ~12 px manchas quase lisas, próximas do marmorizado; `h`/`n` afinam/engrossam durante a execução (passo 1 px, de 1 a 16), e o valor aparece no overlay (`feixe 2px`). Mudar a espessura refaz as cerdas (~15–25 ms naquele quadro) sem desfazer o borrão já feito. Com `--gain 1` a tinta anda a 40% da velocidade do corpo (`LANDSCAPE_FLOW_SCALE`): a pleno, sete pessoas deixavam a pintura irreconhecível em ~5 s; assim, depois de 28 s de ensaio ela continua reconhecível, com ~1/3 da área repintada. A força do borrão começa em `--gain` e muda durante a execução com `s` (diminui) e `x` (aumenta), as mesmas teclas do ganho no xadrez: passo 0,1, entre 0 (a imagem congela) e 5; o valor atual aparece no overlay (`Paisagens: forca 1.0`) e, no `--pi`, no relatório do terminal. As mudanças são permanentes até um reset (trocar de fonte com `m`). **Paletas neon:** `--palette` (ou a tecla `p`, que alterna em ciclo) troca as cores da pintura por uma paleta de estética cyberpunk: `synthwave` (violeta → rosa → laranja), `cyberpunk` (violeta → magenta → ciano → amarelo), `matrix` (verde digital), `tron` (azul elétrico) ou `vaporwave` (lilás, rosa e ciano); `original` mantém as cores da pintura. É um mapa de gradiente: o brilho de cada pixel vira uma cor da paleta, do escuro ao claro, e o brilho de cada pintura é esticado entre os percentis 2 e 98 para ocupar o gradiente todo (as pinturas são escuras). A paleta é aplicada no desenho, não na tela guardada: trocar é instantâneo e mantém o borrão, a regeneração e a janela da silhueta, que também fica neon. As paletas ficam em `LANDSCAPE_PALETTES` (cores de parada, do escuro ao claro), fáceis de editar. Custo: ~0,5 ms por quadro em 640×480 e ~1,4 ms em 720p. **Regeneração:** onde ninguém age há `--regen-delay` segundos (padrão 3), a pintura original volta aos poucos, num cross-fade, e fica completa em ~`--regen-time` segundos (padrão 20); onde os dançarinos continuam agindo, ela não volta. O tempo parado é contado por região, na grade do fluxo óptico (1/8 do quadro), e a volta entra em 1 s, sem degrau. Durante a execução, `d`/`c` diminuem/aumentam a espera (passo 1 s, de 0 a 60) e `f`/`v` o tempo de volta (passo 5 s, de 2 a 300); os valores aparecem no overlay (`volta 3s/20s`). A mistura é aplicada a cada 8 quadros com o passo acumulado, em 16 bits e com um só arredondamento; o último resíduo (onde o passo já não move o pixel em 8 bits) é copiado do original, então a pintura volta exatamente ao original. Medido com o ensaio: depois de 10 s de dança e com a cena parada, ~18 s até a tela ficar idêntica ao original. Custo médio: ~0,4 ms por quadro em 640×480 (1 thread) e ~1,1 ms em 720p; o quadro em que a mistura roda fica ~3 ms (640×480) ou ~9 ms (720p) mais lento. **Silhueta:** `--silhouette` (ou a tecla `g`, como no xadrez) abre, dentro dos corpos, uma janela para a próxima pintura da lista, sem borrão; com uma imagem só, a janela mostra a própria pintura original, sem as pinceladas. É ao vivo: a tela borrada por baixo não muda, e quando o corpo sai a pintura atual reaparece como estava. A máscara é a mesma do xadrez — profundidade no Kinect (`b` calibra o fundo) e MOG2 nas outras fontes. Com MOG2 a silhueta é parcial: ele só separa o que se move do fundo, e quem já estava em cena quando ele começou a aprender demora a aparecer; no ensaio de teste, só parte dos corpos abriu a janela. No perfil normal a borda é macia (~0,7 ms a mais em 640×480, ~3 ms em 720p, onde a mistura é em resolução cheia); no `--pi` é dura (~0,1 ms). Os mapas do `remap` são calculados com operações do OpenCV (`LANDSCAPE_SMEAR_BACKEND = "opencv"`); a versão original em NumPy continua no código e volta trocando a constante para `"numpy"`. O OpenCV arredonda um pouco diferente (até ~1e-4 px nos mapas): num trecho de ensaio de 400 quadros, 13 de 921 600 pixels terminaram 1 nível (de 255) diferentes. Medido com 1 thread: em 1280×720, ~15 ms por quadro com OpenCV contra ~18 ms com NumPy (o borrão em si: 11,7 contra 14,4 ms); em 640×480 (`pi_benchmark.py --mode paisagens`), 4,5 contra 5,4 ms. O resto do quadro é o Farneback (~2 ms em 720p) e conversões.
- **Xadrez:** inspirado nas camisas Madras — faixas de cores diferentes que se cruzam formando quadros de tamanhos desiguais. Ao criar o processador (antes do primeiro frame) são sorteados um sett e a fase de cada direção: faixas de 1 a 6 linhas paralelas do mesmo tom e espessura (1–3 px), separadas por vãos que alternam entre pequenos e grandes; o sett é espelhado, como nos tecidos, repetido e usado nas duas direções de linhas. As cores partem de um único matiz base (`--hue`, em graus: 0 vermelho, 120 verde, 240 azul; padrão 30°, laranja) e variam só em luminosidade: cada faixa sorteia um de 6 níveis de mistura com branco, da cor pura ao branco. `--density` (padrão 1) divide faixas e vãos: valores maiores geram mais linhas e quadros menores. `--rotation` (graus, padrão 0) gira o padrão em torno do centro do quadro; as linhas que não cruzam o quadro são descartadas, então a 45° há mais linhas que a 0°. Durante a execução, `a`/`z` diminuem/aumentam a densidade (fator 1,1, entre 0,25 e 4; até 2 no mix), `s`/`x` o ganho (passo 0,1, entre 0 e 5), `d`/`c` a rotação (passo 5°), e `f`/`v` giram o matiz no disco de cores, anti-horário/horário (passo 10°; horário = vermelho → amarelo → verde → azul); os valores atuais aparecem no overlay e no terminal. Densidade e rotação recriam a malha com o mesmo sett e as mesmas fases — o mesmo tecido, ampliado ou girado — e zeram o movimento; ganho e matiz valem no frame seguinte, sem interromper o movimento. Cada linha é uma cadeia de nós a cada 40 px, desenhada como Catmull-Rom (Bézier cúbicas encadeadas, 6 amostras por trecho), com as pontas fixas fora do quadro. O fluxo óptico amostrado em cada nó o empurra em 2D (`--gain`, padrão 1, é a força; `--sensitivity`, padrão 1, teclas `h`/`n`, é quanto movimento basta: divide o mínimo de movimento do fluxo e da mão e multiplica o alcance da mão — veja [docs/xadrez.md](docs/xadrez.md#sensibilidade)); uma mola o devolve ao repouso e a tensão entre vizinhos faz a linha se comportar como uma corda esticada. Linhas anti-aliased grossas custam ~3× as finas no OpenCV, então as grossas são desenhadas com 1 px numa camada, engrossadas com `cv2.dilate` e compostas com `cv2.max` (~2,8× mais rápido). Medido em 1280×720 com vídeo sintético, sem GPU: 7–11 ms por frame com 60–107 linhas (densidade 1, rotação 0) e 10–14 ms com 100–179 linhas (densidade ~1,7); o número de linhas varia com o sett sorteado. No pior caso (densidade 4 a 45°, ~640 linhas) o custo chega a ~27 ms. **Pulsos de neon** percorrem as linhas como informação. Por padrão nascem das mãos (`--pulse-source hands`): o YOLOv8-pose acha os punhos de cada pessoa numa thread própria e, quando uma mão em movimento passa por uma linha, nascem ali dois pulsos, um para cada lado; limites por mão, por linha e no total (12 linhas/s) evitam a inundação, e sem o YOLO instalado eles voltam a ser sorteados (detalhes em [docs/xadrez.md](docs/xadrez.md#pulsos-de-neon)). Com `--pulse-source random`, nascem em intervalos aleatórios (exponenciais, em média `--pulses` por segundo, padrão 4; 0 desliga), numa linha e ponta sorteadas, e cada um leva de 1 a 6 s (sorteado) para atravessá-la, em tempo real, independente do FPS. A posição é medida ao longo da curva, então o pulso acompanha a linha mesmo deformada pelo movimento. Cada pulso tem uma cauda que apaga para trás, com núcleo claro (matiz misturado a branco) e halo do matiz puro; núcleo e halo são somados ao quadro, então só clareiam. O halo é desenhado a 1/4 da resolução, borrado e ampliado. Todas as caudas são amostradas de uma vez em NumPy e cada nível de brilho é um único `cv2.polylines`: ~2,5 ms com ~15 pulsos vivos (o padrão), ~8 ms no limite de 150. **Silhueta:** ligada por padrão no xadrez (`--no-silhouette` desliga; a tecla `g` liga e desliga) preenche os corpos com a cor pura do matiz base, por baixo das linhas e dos pulsos (no modo `paisagens` a mesma opção abre uma janela para outra pintura; veja acima). Com o Kinect a silhueta vem da máscara de profundidade (`--min-depth`/`--max-depth`; no xadrez, `b` também calibra o fundo); nas outras fontes, de um MOG2 em 1/4 da resolução, que só separa o que se move do fundo — quem fica parado por muito tempo é absorvido (histórico de 500 frames). A máscara é limpa, suavizada, colorida em 1/4 e ampliada com interpolação linear, o que dá bordas macias por ~2 ms (Kinect) ou ~4 ms (MOG2) em 1280×720. As linhas no tom puro do matiz se confundem com o preenchimento dentro do corpo; as mais claras continuam visíveis. **Disposição:** `--layout mix` (padrão, inclusive no `--pi`) sobrepõe três malhas independentes; `--layout regular` (ou a tecla `l`, que alterna) usa uma malha só. No mix a densidade é limitada a 2 (`PLAID_MIX_MAX_DENSITY`): um `--density` maior é reduzido com um aviso, `z` não passa de 2 e entrar no mix pela tecla `l` reduz a densidade se preciso (ela não volta ao valor anterior ao sair). O mix sobrepõe três malhas independentes, cada uma com seu próprio padrão sorteado e desalinhadas entre si, como se estivessem em alturas diferentes em relação à tela. Cada camada além da regular tem um desvio de ângulo sorteado na inicialização, somado à rotação global, a pelo menos 20° (`PLAID_MIX_MIN_ANGLE`) das outras — medido módulo 90°, já que o xadrez se repete a cada 90° —, para que os quadros de cada densidade apareçam como tramas distintas em vez de linhas paralelas que só adensam a malha. Quanto ao resto: a distante tem o dobro da densidade, 40% do brilho, linhas 1 px mais finas e recebe metade do empurrão do fluxo; a do meio é exatamente a malha regular, com 70% do brilho; a próxima tem 0,55× a densidade, brilho pleno, linhas 1 px mais grossas e 1,6× o empurrão (paralaxe: o que está perto se move mais). Densidade, rotação, ganho, matiz, pulsos e silhueta continuam valendo para todas; `--pulses` é o total, dividido entre as camadas pelo número de linhas. As linhas finas são desenhadas da camada distante para a próxima e as grossas de todas as camadas são compostas juntas, uma passada de `dilate` + `max` por espessura (a cor mais clara, da camada próxima, vence). Custo medido em 1280×720: ~10,5 ms com 70 linhas (regular) e 20–27 ms com 300–420 linhas (mix, conforme o sorteio — malhas giradas cruzam o quadro com mais linhas); a silhueta soma ~5 ms em ambos. Os fatores de cada camada ficam em `PLAID_LAYOUTS`.
- **YOLO com GPU:** com CUDA, a rede PyTorch é capturada uma vez como CUDA Graph e reexecutada a cada frame com uma única chamada. Lançar as ~200 camadas a partir do Python custava mais CPU do que a GPU gasta no yolov8n: medido aqui, a inferência caiu de ~15 ms para ~2,8 ms e o modo YOLO foi de ~24 para ~50 fps. Os keypoints são idênticos aos do Ultralytics. Se a captura do grafo falhar, a PoC avisa e usa o Ultralytics em CUDA.
- **YOLO sem GPU:** `--nogpu` prefere OpenVINO/CPU, chamado diretamente (sem o pré/pós-processamento do Ultralytics), com um modelo exportado em entrada retangular 256×320: o frame 320×240 recebe só 16 linhas de borda, em vez de 80 no modelo quadrado. Os keypoints são idênticos aos do Ultralytics com o mesmo modelo. Se o modelo não existir, a PoC exporta o `.pt` automaticamente. Se o INT8 não puder ser exportado ou carregado, tenta o FP32; se OpenVINO ou as exportações falharem, usa PyTorch CPU e avisa no terminal. Sem `--nogpu`, CUDA disponível tem prioridade.
- **INT8 (padrão em CPU):** `--nogpu` usa `yolov8n-pose_256x320_int8_openvino_model/`, quantizado com NNCF; `--no-int8` volta ao FP32 `yolov8n-pose_256x320_openvino_model/`. Medido neste PC: ~40% mais FPS que o FP32 (14 → 20 fps), mas os keypoints tremem mais (≈1 px a mais de variação frame a frame, na escala 320, e até ~4 px no p90 em 1280×720) — compense com `--point-deadband` maior ou use `--no-int8` se necessário. Com CUDA, `--int8` é ignorado. A exportação automática calibra com o `coco8-pose.yaml` do Ultralytics (baixa ~1 MB e instala `nncf` na primeira vez); para calibrar com frames dos ensaios, chame `export_yolo_openvino_model(int8=True, calibration_data='dados.yaml')`.
- **Kinect:** máscara RAW pela faixa de profundidade, com calibração de fundo opcional pela tecla `b`.

As metas de latência (<8 ms MOG2, <5 ms Farneback, <20 ms YOLO) são metas por algoritmo, não garantias de latência ponta a ponta. A janela debug/projetores e o FPS de captura também consomem tempo; valide com `kinect_diagnostic.py`/telemetria no hardware final.

MP4, MOV e M4V são reproduzidos em loop. Pressione `m` para alternar, sem fechar o programa, entre Kinect, webcam e o vídeo configurado (por `--source` ou `--fallback-video`). Se o Kinect selecionado não iniciar, a aplicação tenta webcam e depois o vídeo informado em `--fallback-video`. Pressione `q` para sair. Para remover paredes e outros objetos estáticos da máscara no Kinect, deixe a cena vazia e pressione `b` na janela de debug. A faixa inicial de profundidade vem de `--min-depth` e `--max-depth`, em mm (padrão: 800 a 3000):

```bash
python dance_interactive_poc.py --min-depth 1000 --max-depth 2500
```

No Kinect, os atalhos alteram a faixa em passos de 100 mm:

| Tecla | Ação |
| --- | --- |
| `a` | Reduz o limite mínimo |
| `z` | Aumenta o limite mínimo |
| `s` | Reduz o limite máximo |
| `x` | Aumenta o limite máximo |
| `b` | Captura/recria o fundo; faça isso com a área de dança vazia |

Esses ajustes só se aplicam ao Kinect. Em webcam/vídeo não há mapa de profundidade; o algoritmo escolhido produz a imagem do corpo.

No modo `xadrez`, as mesmas teclas (e `d`/`c`) ajustam o padrão:

| Tecla | Ação |
| --- | --- |
| `a` / `z` | Diminui/aumenta a densidade (fator 1,1, entre 0,25 e 4; até 2 no `mix`) |
| `s` / `x` | Diminui/aumenta o ganho (passo 0,1, entre 0 e 5) |
| `h` / `n` | Diminui/aumenta a sensibilidade da interação dos corpos com as linhas (×1,25, entre 0,25 e 4) |
| `d` / `c` | Gira o padrão −5°/+5° |
| `f` / `v` | Gira o matiz no disco de cores, anti-horário/horário (−10°/+10°) |
| `g` | Liga/desliga a silhueta dos corpos na cor base |
| `l` | Alterna a malha entre `mix` (padrão, malhas em alturas diferentes) e `regular` |
| `b` | Com Kinect, calibra o fundo da máscara usada pela silhueta |

**Metrônomo (todos os modos):** `--metronome` liga um metrônomo em compasso 4/4, com a cabeça do compasso mais aguda (1760 Hz) e mais forte que os outros tempos (880 Hz); `--bpm` define o andamento (padrão 60, entre 30 e 240). Os cliques são sintetizados dentro do callback de áudio do `sounddevice` e posicionados na amostra exata, então o andamento não depende do FPS do vídeo nem das threads de processamento; ligar, desligar e mudar o andamento valem a partir do próximo bloco de áudio (dezenas de ms). Ao ligar, o compasso recomeça pela cabeça; ao mudar o andamento, o próximo tempo já segue o novo valor, sem reiniciar o compasso. O estado aparece no overlay de debug e, no `--pi`, no relatório do terminal. Sem a PortAudio, a PoC avisa e segue sem som. O áudio sai pelo dispositivo padrão do sistema.

| Tecla | Ação |
| --- | --- |
| `t` | Liga/desliga o metrônomo |
| `-` | Diminui o andamento (−5 bpm) |
| `=` ou `+` | Aumenta o andamento (+5 bpm) |

**Calibração do fundo Kinect:** ao pressionar `b`, a PoC registra a mediana de 15 frames (cerca de 0,5 s) como fundo estático. A máscara passa a incluir somente pixels dentro de `Min Depth`–`Max Depth` que estejam pelo menos 80 mm mais próximos que o fundo capturado. A parede fica fora da máscara e o dançarino, entre a câmera e a parede, aparece. O overlay mostra `Fundo: calibrado` quando ativo. Faça a captura sem pessoas na área e repita se mover a câmera, a parede ou objetos grandes; se o dançarino estiver presente durante a calibração, ele será tratado como fundo e sumirá da máscara.

Sem calibrar o fundo, a máscara continua sendo apenas um corte por distância e **tudo** dentro da faixa aparece, incluindo parede, chão e móveis. O mapa usa `0` para "sem leitura"; nesta unidade, leituras válidas foram observadas desde ~410 mm. A "sombra" junto ao corpo vem da oclusão do padrão infravermelho do Kinect; pixels sem leitura continuam fora da máscara. A diferença de 80 mm pode ser insuficiente para superfícies que se moveram pouco, ou excessiva para partes muito finas do corpo; os limites Min/Max continuam sendo aplicados.

Todos os capturadores entregam `(success, frame_bgr, depth, body_mask)`. Webcam e vídeo devolvem `None` em `depth` e `body_mask`; o algoritmo selecionado produz a máscara. O vídeo é decodificado com 2 threads do FFmpeg, que bastam para 720p60 e deixam núcleos livres para a inferência. `Ctrl+C` também solicita o encerramento.

## Janelas e telemetria

- **Debug & Tracking:** vídeo da fonte com máscara, vetores ou caixa/esqueleto conforme o algoritmo, FPS, latência, modo e fonte ativos.
- **Projetor Corpo/Frontal:** MOG2/Kinect projetam a máscara vermelha; Optical Flow projeta vetores; Conexões projeta a malha interpessoal sem suavização em fundo preto; Capim, Mato e Xadrez projetam o capim, o mato ou o xadrez (o mato cobre a tela inteira); Paisagens projeta a imagem sendo repintada.

`--fullscreen` abre a janela do projetor em tela cheia, sem bordas nem barra de título, no monitor em que ela estiver; `--no-fullscreen` a mantém como janela comum. Sem o argumento, o padrão é tela cheia apenas no `--pi` (que roda ligado ao projetor) e janela comum no perfil normal, para não cobrir a janela de debug. A janela de debug nunca vai a tela cheia. Mova a janela do projetor para o monitor/projetor desejado antes de alternar — o OpenCV usa o monitor atual da janela.

As janelas Chão/Fundo e Simulação/Consolidada estão temporariamente desativadas; o loop não calcula centroide, ondas nem composição dessas telas.

A latência exibida é medida por `time.perf_counter()` desde o timestamp associado ao frame até o fim do ciclo de exibição/`waitKey`. É uma estimativa de software; não mede exposição do sensor, sincronização real dos projetores ou o tempo até o conteúdo aparecer fisicamente. Os timestamps de Kinect e webcam também não são equivalentes, portanto os resultados entre modos não devem ser comparados como uma medição calibrada.

O indicador usa verde até 35 ms, amarelo acima de 35 até 45 ms e vermelho acima de 45 ms. O FPS representa a taxa observada pelo loop visual, não necessariamente a taxa nativa do sensor.

## Raspberry Pi 4 (`--pi`)

`--pi` é um perfil para rodar com Kinect num Raspberry Pi 4 (4 GB). Aceita `--mode xadrez` (o padrão) ou `--mode paisagens` (outros modos são recusados). Em ambos só a janela do projetor é aberta; no `paisagens` é isso que muda, já que ele não desenha linhas. No `xadrez`, a qualidade de desenho passa a `PI_PLAID_RENDER`:

- **Sem anti-aliasing** em linhas e pulsos (`cv2.LINE_8`); com isso as linhas grossas saem direto do `cv2.polylines`, sem a camada + `dilate` + `max`.
- **Curvas com 3 amostras por trecho** em vez de 6: as linhas só curvam onde o fluxo as empurra, e a diferença quase não aparece.
- **Silhueta com borda dura**: sem o desfoque; a máscara é ampliada em 1 canal e recorta um quadro de cor sólida (~30% mais barato que colorir e ampliar 3 canais).
- **No máximo 40 pulsos** ao mesmo tempo (150 no perfil normal).
- **Só a janela do projetor**: sem janela de debug, overlay e composição câmera + desenho; FPS, latência e ajustes atuais vão para o terminal a cada 5 s. As teclas continuam funcionando na janela do projetor.

Todo o resto (algoritmo, teclas, `--density`, `--gain`, `--rotation`, `--hue`, `--pulses`, `--silhouette`, `--layout`) funciona igual. O Kinect já entrega RGB em 640×480, então nada é reduzido além disso.

```bash
python -m pip install -r requirements-pi.txt -r requirements-kinect.txt
python dance_interactive_poc.py --pi --silhouette
python pi_benchmark.py            # mede os perfis sem câmera nem janelas
python pi_benchmark.py --display  # inclui o imshow em tela cheia
python pi_benchmark.py --mode paisagens  # compara os mapas NumPy e OpenCV do borrão
```

**Medido (i7-4790, 640×480, OpenCV limitado a 1 thread, `pi_benchmark.py`), só o processamento:**

| Perfil | Malha | Silhueta | Linhas | ms/quadro |
| --- | --- | --- | ---: | ---: |
| normal | regular | não / sim | 83 | 5,9 / 6,9 |
| `--pi` | regular | não / sim | 83 | 4,9 / 5,6 |
| normal | mix | sim | 301 | 13,7 |
| `--pi` | mix | sim | 301 | 12,3 |

Em 640×480 o anti-aliasing pesa bem menos que em 720p: o ganho do `--pi` neste PC é de ~15–20%. O que sobra está dividido entre desenhar as linhas (~1,7 ms), redimensionamentos (~1 ms), Farneback (~0,6 ms) e a física/curvas em NumPy (~0,8 ms). **Estimativa, não medida:** um núcleo do Pi 4 deve ser 4–6× mais lento que este, o que dá ~22–34 ms por quadro em `regular` com silhueta (perto dos 30 fps do Kinect) e ~50–75 ms em `mix` (~15 fps). Processamento, captura e exibição rodam em threads separadas, então o FPS fica limitado pela etapa mais lenta, não pela soma. Rode `pi_benchmark.py` no Pi para trocar a estimativa por números.

**Recomendações para o Pi:**

- **Resolução da saída:** o `imshow` em tela cheia amplia o quadro na CPU. Configure a saída HDMI do Pi em 640×480 ou 800×600 (o projetor amplia de graça) — é o maior custo ainda não medido.
- **Sem desktop:** Raspberry Pi OS Lite (64 bits) com uma sessão X mínima libera CPU e memória do compositor.
- **Resfriamento:** o Pi 4 reduz o clock perto de 80–85 °C; use dissipador com ventoinha (o mesmo problema que distorcia as medições do PC de desenvolvimento).
- **`mix` (o padrão):** pela estimativa fica em ~15 fps no Pi; se precisar de 30, reduza a densidade da camada distante em `PLAID_LAYOUTS` (ela tem mais da metade das linhas), baixe `--density` ou use `--layout regular` / a tecla `l`.

**Alternativas avaliadas:**

| Alternativa | Ganho esperado | Custo |
| --- | --- | --- |
| Saída HDMI em 640×480/800×600 | Elimina a ampliação na CPU do `imshow` | Configuração do Pi |
| Exibição via SDL2/pygame com escala na GPU (KMS/DRM, sem desktop) | Ampliação e envio do quadro na GPU; CPU livre para o processamento | Nova dependência e um caminho de exibição alternativo |
| Desenho em OpenGL ES 3.1 (VideoCore VI): CPU só calcula fluxo e física, a GPU desenha linhas, pulsos e brilho | O maior: anti-aliasing e `mix` a 30 fps ficariam ao alcance | Reescrita do desenho em shaders |
| Desenhar em 320×240 e ampliar | ~4× menos pixels desenhados | Linhas grossas e serrilhadas; só vale com a ampliação na GPU |
| Atualizar o fluxo a 15 Hz | ~0,6 ms (Farneback) a cada dois quadros | Resposta ao movimento menos imediata |
| Fluxo na imagem de profundidade em vez do RGB | Pouco em custo; ganha robustez | Ver a ressalva abaixo |

**Ressalva importante para a instalação:** o Farneback roda no RGB do Kinect, e a câmera RGB vai ver a própria projeção (linhas e pulsos em movimento sobre os corpos e o fundo). Isso pode realimentar o fluxo: o desenho empurrando o próprio desenho. A profundidade do Kinect é infravermelha e não enxerga a projeção; calcular o fluxo sobre ela (ou sobre a máscara de profundidade) evitaria esse efeito. Vale testar no espaço real antes de qualquer outra otimização.

## Desempenho medido

Comparação entre `v0.17.1` e `v0.18.2`, no PC de desenvolvimento (Intel 4 núcleos/8 threads, RTX 3060), alternando as versões e esperando a CPU esfriar entre execuções de 15 s; médias de 2 rodadas. `videos/MVI_1741.MOV` é 720p a 60 fps com ~6 pessoas por quadro; o Kinect entrega 640×480 a 30 fps, que limita o FPS. Latência = mediana do tempo entre a captura e o fim do `waitKey`.

| Modo | v0.17.1 | v0.18.2 |
| --- | --- | --- |
| YOLO CUDA, vídeo | 18,0 fps · 58 ms | 53,3 fps · 18 ms |
| YOLO `--nogpu` FP32 (hoje `--no-int8`), vídeo | 8,3 fps · 127 ms | 13,6 fps · 78 ms |
| YOLO `--nogpu` INT8 (padrão desde v0.19.0), vídeo | — | 23,5 fps · 49 ms |
| YOLO CUDA, Kinect | 30 fps · 23 ms | 30 fps · 14 ms |
| YOLO `--nogpu` FP32, Kinect | 21,8 fps · 58 ms | 27,4 fps · 28 ms |
| MOG2, vídeo (`videos/MVI_1724.MOV`) | 36,0 fps · 37 ms | 47,9 fps · 30 ms |
| Kinect (profundidade) | 30 fps · 9 ms | 30 fps · 16 ms |

No modo Kinect puro, a latência medida subiu porque a v0.17.1 reenviava a mesma imagem ao Qt continuamente (≈15 `imshow` por frame, um núcleo inteiro ocupado); manter esse reenvio baixa o número medido para ~9 ms, mas piora o YOLO e não há como saber, sem medir a projeção, se a imagem aparece antes. A decisão final depende de uma medição física (roadmap, item 2).

**Resfriamento:** essas medições foram feitas com o cooler original da CPU com defeito — ela chegava a 100 °C em segundos e o kernel injetava pausas (`intel_powerclamp`, até 49% em todos os núcleos), o que limitava sobretudo os modos que dependem da CPU. Depois da troca do cooler (08/10/2026), 15 min em carga total estabilizam em 84–86 °C sem pausas, e os mesmos benchmarks (médias de 2 rodadas, sem Kinect conectado) ficaram assim:

| Modo (vídeo) | v0.18.2, cooler antigo | v0.18.2, cooler novo | v0.23.0, cooler novo |
| --- | --- | --- | --- |
| YOLO CUDA | 53,3 fps · 18 ms | 59,9 fps · 17 ms | 59,9 fps · 17 ms |
| YOLO `--nogpu` FP32 | 13,6 fps · 78 ms | 49,6 fps · 27 ms | 46,0 fps · 29 ms |
| YOLO `--nogpu` INT8 | 23,5 fps · 49 ms | 54,6 fps · 26 ms | 53,0 fps · 26 ms |
| MOG2 (`videos/MVI_1724.MOV`) | 47,9 fps · 30 ms | 59,9 fps · 12 ms | 59,9 fps · 12 ms |

59,9 fps é o limite dos vídeos (60 fps). A pequena perda da v0.23.0 em CPU vem do desenho das linhas mais grossas (até 20 px).

## Versionamento e commits

O projeto segue [Versionamento Semântico](https://semver.org/lang/pt-BR/) e [Conventional Commits](https://www.conventionalcommits.org/pt-br/). A branch principal é `main` e cada versão publicada recebe uma tag anotada `vX.Y.Z`.

| Tipo de commit | Efeito na versão |
| --- | --- |
| `feat:` | minor (`0.X.0`) |
| `fix:`, `perf:` | patch (`0.0.X`) |
| `feat!:` ou rodapé `BREAKING CHANGE:` | major (`X.0.0`); antes da `1.0.0`, incrementa o minor |
| `docs:`, `chore:`, `refactor:`, `test:`, `build:`, `ci:` | sem mudança de versão |

Exemplo: `feat(projection): add floor and body composite preview`. Enquanto a versão for `0.y.z`, a API ainda é instável.

## Roadmap

1. **Validar hardware e instalação:** testar Kinect 1414/1473, webcam, arquivos de vídeo em loop e encerramento em cada backend.
2. **Calibrar latência:** documentar timestamp de captura em cada sensor, separar idade do frame de processamento/renderização e medir com ferramenta externa a resposta física do projetor.
3. **Melhorar interação:** ajustar o limiar da subtração de fundo e tratar sombras/oclusões, estabilizar o centro de massa, detectar pés/partes do corpo e disparar eventos de onda por movimento.
4. **Preparar projeção:** permitir configuração de resolução, posição das janelas, fullscreen por projetor e calibração geométrica.
5. **Consolidar operação:** adicionar testes automatizados para funções sem hardware e registrar FPS/latência em execuções longas.

## Limitações conhecidas

- `--fullscreen` só escolhe tela cheia ou janela: o monitor usado é o do sistema de janelas e ainda não há posicionamento automático nem calibração de projetores.
- A medição por modo depende da câmera, resolução, CPU/GPU e backend de exibição; as metas de latência ainda precisam ser medidas no hardware de apresentação.
- A combinação de sistema operacional, driver, câmera, resolução e projetor determina a latência real.

### Kinect: avisos USB e áudio

`Could not open audio: -1` refere-se ao dispositivo/canal de áudio, que esta PoC não utiliza para RGB ou profundidade. Já mensagens como `USB device disappeared`, `control transfer failed` e `USB camera marked dead` indicam falha de comunicação com o Kinect. Verifique cabo e alimentação, conexão direta a uma porta USB 2.0, permissões udev e se outro processo está usando o Kinect. Se a thread nativa continuar bloqueada após o pedido de parada, ela é daemon e não impede a tentativa de fallback nem a saída do processo; porém, a libfreenect pode continuar emitindo logs até o processo encerrar.

**Partida lenta é normal neste equipamento:** o primeiro frame do Kinect chega em ~6 a 8 s depois de abrir o dispositivo (medido; depois disso a captura mantém 30 fps). A PoC espera até 20 s antes de considerar o Kinect indisponível. Um `Lost N total packets` isolado no início do stream e um `Expected 1748 data bytes` ocasional são esperados e não indicam defeito; o sinal de problema é perda contínua, `resyncing` ou `control transfer failed` repetidos.

Na inicialização, se o Kinect falhar, a PoC tenta webcam e em seguida `--fallback-video`, se informado. Durante a execução, `m` alterna entre as fontes configuradas. Se webcam e vídeo de fallback também falharem, a PoC informa os erros. Confirme a presença da câmera com `ls /dev/video*` e teste o Kinect independentemente com `freenect-glview`.

Para um veredito objetivo sobre o Kinect, rode `python kinect_diagnostic.py [segundos]` (padrão: 15 s medidos após o primeiro frame). Ele lista os dispositivos USB com velocidade e controlador, captura profundidade/RGB com `runloop` em um subprocesso (que pode ser encerrado se travar) e informa o tempo até o primeiro frame, o FPS em regime estável, o maior intervalo entre frames e as perdas registradas pela libfreenect. Retorna `0` (OK), `1` (instável) ou `2` (falha). Se o veredito for instável, repita após trocar de porta ou cabo; o Kinect v1 às vezes é mais estável em portas EHCI (`ehci-pci` no `lsusb -t`) do que em xHCI (`xhci_hcd`), mas neste equipamento a captura foi estável em xHCI.
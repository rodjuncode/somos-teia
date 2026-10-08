# Dança Interativa: PoC

Prova de conceito em Python para explorar visuais generativos dirigidos pelo movimento, usando Kinect v1 (1414/1473), webcam USB ou arquivo MP4/MOV.

## Estado atual

O protótipo em [`dance_interactive_poc.py`](dance_interactive_poc.py) já implementa:

- Captura contínua em thread separada, mantendo o frame mais recente disponível para o loop visual.
- Três fontes intercambiáveis pela CLI e pela tecla `m`: Kinect v1, webcam USB e arquivo MP4/MOV em loop.
- Fallback automático de Kinect para webcam e, opcionalmente, arquivo de vídeo.
- Saída padronizada de todos os capturadores: sucesso, frame RGB, profundidade real ou matriz dummy e máscara corporal.
- Webcam e vídeo usam modos selecionáveis: MOG2, fluxo óptico Farneback ou YOLOv8n-pose. O padrão é MOG2, sem inferência de IA.
- Modo Kinect mantém profundidade RAW; se a inicialização falhar, o fallback usa MOG2 na webcam/vídeo.
- Máscara binária por faixa de profundidade no Kinect, configurada inicialmente entre 800 e 3000 mm (ajustável por linha de comando e por atalhos).
- Overlay de máscara/contornos, vetores ópticos ou esqueleto YOLO, FPS, modo ativo e latência estimada.
- Duas janelas OpenCV neste momento: debug e projetor Corpo/Frontal. As telas de Chão/Fundo e Simulação/Consolidada e todo o cálculo associado estão temporariamente removidos.
- Encerramento de captura e janelas ao pressionar `q`.

O código compila e MOG2, Farneback e YOLO foram verificados com quadros sintéticos; YOLO real/model weights e execução ponta a ponta nos projetores ainda não foram validados nesta revisão. As metas de latência são objetivos por algoritmo, não garantias ponta a ponta.

## Requisitos

- Linux/Ubuntu, Python 3 e `python3-venv`.
- Para Kinect v1: dispositivo conectado, bibliotecas de desenvolvimento `libfreenect` e binding Python `freenect`.
- Para webcam/fallback: câmera acessível e dependências Python do [`requirements.txt`](requirements.txt).

O Kinect requer a biblioteca nativa `libfreenect` e headers de desenvolvimento; a binding Python e as demais dependências ficam isoladas no venv. O pacote `python3-freenect` não é usado. Ultralytics/PyTorch adiciona dependências grandes e baixa `yolov8n-pose.pt` ao primeiro uso do modo YOLO.

```bash
sudo apt-get update
sudo apt-get install libfreenect-dev libfreenect-bin python3-dev python3-venv build-essential
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-kinect.txt
```

O comando pip instala as dependências de visão dentro do venv. Os pacotes estão listados em [`requirements.txt`](requirements.txt); `requirements-kinect.txt` também instala a binding `freenect` e Cython. Para YOLO, use `--mode yolo`; MOG2 e Farneback não precisam de rede neural.

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
python dance_interactive_poc.py --source ensaio.mp4 --mode yolo
python dance_interactive_poc.py --source ensaio.mp4 --mode yolo --nogpu
python dance_interactive_poc.py --source ensaio.mp4 --mode yolo --nogpu --int8
python dance_interactive_poc.py --source ensaio.mp4 --mode yolo --max-distance 150 --max-connections 5 --point-deadband 4
python dance_interactive_poc.py --source ensaio.mp4 --mode yolo --show-points
python dance_interactive_poc.py --source kinect --mode kinect
```

Para fontes webcam/vídeo, o modo padrão é `mog2`. `--mode kinect` exige `--source kinect`; os outros modos podem processar RGB de webcam, vídeo ou Kinect. `m` continua alternando as fontes disponíveis sem reiniciar e reseta o estado temporal do algoritmo quando necessário.

- **MOG2:** `BackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)` roda em 160×120; a máscara é ampliada ao tamanho original, limpa com abertura/fechamento morfológicos e seu centroide vem dos momentos.
- **Optical Flow:** Farneback roda em 80×60 entre frames cinza consecutivos; a máscara seleciona movimento acima de 0,15 px nessa escala (~1,2 px na imagem original) e a janela Corpo mostra vetores ampliados e recortados pela máscara.
- **YOLO:** `yolov8n-pose.pt`, inferência em `imgsz=320` com até 16 detecções por frame para limitar o pós-processamento; cada dançarino gera sete nós: cabeça (média de nariz/olhos/orelhas) e um nó por classe bilateral (ombro, cotovelo, pulso/mão, quadril, joelho e tornozelo), usando o ponto esquerdo quando visível e o direito como fallback, sem criar um ponto médio artificial. Os círculos dos nós ficam ocultos por padrão; `--show-points` os habilita. `--point-deadband` (padrão 4 px) ignora tremor pequeno sem EMA; movimentos maiores avançam no mesmo frame. As distâncias entre os nós que ainda têm conexões livres são calculadas de uma vez em NumPy, respeitando `--max-distance` (padrão 150 px), e as arestas mais próximas são escolhidas até `--max-connections` por nó (padrão 5). Com no máximo 112 nós (16 dançarinos × 7), essa busca vetorizada mede 0,1–0,6 ms, de 5× a 80× mais rápida que o quadtree em Python usado antes. Conexões são persistentes: depois de criada, uma aresta não é recalculada — o desenho apenas acompanha os dois nós, mesmo que se afastem além de `--max-distance` — e só é liberada quando um dos nós deixa de ser detectado; a busca considera apenas nós com conexões livres. Brilho e espessura são definidos pela proximidade no momento em que a conexão nasce; os segmentos rígidos YOLO são ignorados.
- **YOLO sem GPU:** `--nogpu` prefere OpenVINO/CPU, chamado diretamente (sem o pré/pós-processamento do Ultralytics), com um modelo exportado em entrada retangular 256×320: o frame 320×240 recebe só 16 linhas de borda, em vez de 80 no modelo quadrado. Os keypoints são idênticos aos do Ultralytics com o mesmo modelo. Se `yolov8n-pose_256x320_openvino_model/` não existir, a PoC exporta o `.pt` automaticamente. Se OpenVINO ou a exportação falharem, usa PyTorch CPU e avisa no terminal. Sem `--nogpu`, CUDA disponível tem prioridade.
- **INT8 (opcional):** `--nogpu --int8` usa `yolov8n-pose_256x320_int8_openvino_model/`, quantizado com NNCF. Medido neste PC: ~40% mais FPS que o FP32 (14 → 20 fps), mas os keypoints tremem mais (≈1 px a mais de variação frame a frame, na escala 320, e até ~4 px no p90 em 1280×720) — compense com `--point-deadband` maior se necessário. A exportação automática calibra com o `coco8-pose.yaml` do Ultralytics (baixa ~1 MB e instala `nncf` na primeira vez); para calibrar com frames dos ensaios, chame `export_yolo_openvino_model(int8=True, calibration_data='dados.yaml')`.
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

Esses ajustes só se aplicam ao Kinect. Em webcam/vídeo, o mapa de profundidade é uma matriz zerada e o algoritmo escolhido produz `body_mask`.

**Calibração do fundo Kinect:** ao pressionar `b`, a PoC registra a mediana de 15 frames (cerca de 0,5 s) como fundo estático. A máscara passa a incluir somente pixels dentro de `Min Depth`–`Max Depth` que estejam pelo menos 80 mm mais próximos que o fundo capturado. A parede fica fora da máscara e o dançarino, entre a câmera e a parede, aparece. O overlay mostra `Fundo: calibrado` quando ativo. Faça a captura sem pessoas na área e repita se mover a câmera, a parede ou objetos grandes; se o dançarino estiver presente durante a calibração, ele será tratado como fundo e sumirá da máscara.

Sem calibrar o fundo, a máscara continua sendo apenas um corte por distância e **tudo** dentro da faixa aparece, incluindo parede, chão e móveis. O mapa usa `0` para "sem leitura"; nesta unidade, leituras válidas foram observadas desde ~410 mm. A "sombra" junto ao corpo vem da oclusão do padrão infravermelho do Kinect; pixels sem leitura continuam fora da máscara. A diferença de 80 mm pode ser insuficiente para superfícies que se moveram pouco, ou excessiva para partes muito finas do corpo; os limites Min/Max continuam sendo aplicados.

Todos os capturadores entregam ao loop principal `(success, frame_rgb, depth_or_dummy, body_mask)`. Webcam e vídeo preenchem `depth_or_dummy` com zeros; o algoritmo selecionado produz a máscara. `Ctrl+C` também solicita o encerramento.

## Janelas e telemetria

- **Debug & Tracking:** vídeo da fonte com máscara, vetores ou caixa/esqueleto conforme o algoritmo, FPS, latência, modo e fonte ativos.
- **Projetor Corpo/Frontal:** MOG2/Kinect projetam a máscara vermelha; Optical Flow projeta vetores; YOLO projeta a malha interpessoal sem suavização em fundo preto.

As janelas Chão/Fundo e Simulação/Consolidada estão temporariamente desativadas; o loop não calcula centroide, ondas nem composição dessas telas.

A latência exibida é medida por `time.perf_counter()` desde o timestamp associado ao frame até o fim do ciclo de exibição/`waitKey`. É uma estimativa de software; não mede exposição do sensor, sincronização real dos projetores ou o tempo até o conteúdo aparecer fisicamente. Os timestamps de Kinect e webcam também não são equivalentes, portanto os resultados entre modos não devem ser comparados como uma medição calibrada.

O indicador usa verde até 35 ms, amarelo acima de 35 até 45 ms e vermelho acima de 45 ms. O FPS representa a taxa observada pelo loop visual, não necessariamente a taxa nativa do sensor.

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

- OpenCV pode abrir as duas janelas de projetor em tela cheia no mesmo monitor; a disposição depende do sistema de janelas e ainda não há calibração de projetores.
- A medição por modo depende da câmera, resolução, CPU/GPU e backend de exibição; as metas de latência ainda precisam ser medidas no hardware de apresentação.
- A combinação de sistema operacional, driver, câmera, resolução e projetor determina a latência real.

### Kinect: avisos USB e áudio

`Could not open audio: -1` refere-se ao dispositivo/canal de áudio, que esta PoC não utiliza para RGB ou profundidade. Já mensagens como `USB device disappeared`, `control transfer failed` e `USB camera marked dead` indicam falha de comunicação com o Kinect. Verifique cabo e alimentação, conexão direta a uma porta USB 2.0, permissões udev e se outro processo está usando o Kinect. Se a thread nativa continuar bloqueada após o pedido de parada, ela é daemon e não impede a tentativa de fallback nem a saída do processo; porém, a libfreenect pode continuar emitindo logs até o processo encerrar.

**Partida lenta é normal neste equipamento:** o primeiro frame do Kinect chega em ~6 a 8 s depois de abrir o dispositivo (medido; depois disso a captura mantém 30 fps). A PoC espera até 20 s antes de considerar o Kinect indisponível. Um `Lost N total packets` isolado no início do stream e um `Expected 1748 data bytes` ocasional são esperados e não indicam defeito; o sinal de problema é perda contínua, `resyncing` ou `control transfer failed` repetidos.

Na inicialização, se o Kinect falhar, a PoC tenta webcam e em seguida `--fallback-video`, se informado. Durante a execução, `m` alterna entre as fontes configuradas. Se webcam e vídeo de fallback também falharem, a PoC informa os erros. Confirme a presença da câmera com `ls /dev/video*` e teste o Kinect independentemente com `freenect-glview`.

Para um veredito objetivo sobre o Kinect, rode `python kinect_diagnostic.py [segundos]` (padrão: 15 s medidos após o primeiro frame). Ele lista os dispositivos USB com velocidade e controlador, captura profundidade/RGB com `runloop` em um subprocesso (que pode ser encerrado se travar) e informa o tempo até o primeiro frame, o FPS em regime estável, o maior intervalo entre frames e as perdas registradas pela libfreenect. Retorna `0` (OK), `1` (instável) ou `2` (falha). Se o veredito for instável, repita após trocar de porta ou cabo; o Kinect v1 às vezes é mais estável em portas EHCI (`ehci-pci` no `lsusb -t`) do que em xHCI (`xhci_hcd`), mas neste equipamento a captura foi estável em xHCI.
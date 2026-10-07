# Dança Interativa: PoC

Prova de conceito em Python para explorar visuais generativos dirigidos pelo movimento, usando Kinect v1 (1414/1473) ou webcam como fallback.

## Estado atual

O protótipo em [`dance_interactive_poc.py`](dance_interactive_poc.py) já implementa:

- Captura contínua em thread separada, mantendo o frame mais recente disponível para o loop visual.
- Seleção do Kinect v1 quando o `freenect` e o dispositivo estão acessíveis; caso contrário, tenta webcam com MediaPipe Pose e Selfie Segmentation.
- Máscara binária por faixa de profundidade no Kinect, configurada inicialmente entre 500 e 950 mm.
- Overlay de máscara/contornos ou landmarks, FPS, faixa de profundidade e latência estimada.
- Três janelas OpenCV: debug, visual de chão e padrão generativo recortado pela máscara do corpo.
- Encerramento de captura e janelas ao pressionar `q`.

O script passou por compilação sintática no ambiente virtual. Ainda não houve validação com Kinect, webcam, projetores ou medição física de latência. O alvo de menos de 35 ms é um objetivo de desenvolvimento, não uma garantia da PoC.

## Requisitos

- Linux/Ubuntu, Python 3 e `python3-venv`.
- Para Kinect v1: dispositivo conectado, bibliotecas de desenvolvimento `libfreenect` e binding Python `freenect`.
- Para webcam/fallback: câmera acessível e dependências Python do [`requirements.txt`](requirements.txt).

O Kinect requer bibliotecas nativas do sistema e uma binding Python dentro do venv. O pacote apt `freenect` instala a biblioteca nativa e suas dependências; a binding Python é instalada pelo [`requirements-kinect.txt`](requirements-kinect.txt). Não é necessário expor pacotes globais ao ambiente virtual.

```bash
sudo apt-get update
sudo apt-get install libfreenect-dev freenect python3-dev python3-venv build-essential
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Opcional: binding Kinect v1 (requer as bibliotecas nativas acima)
python -m pip install -r requirements-kinect.txt
```

`requirements.txt` contém `opencv-python`, `numpy` e `mediapipe`; `requirements-kinect.txt` inclui esse conjunto e adiciona a binding `freenect`. O build da binding usa Cython conforme os metadados do pacote. Para usar apenas webcam, basta instalar `requirements.txt` e pular os pacotes apt específicos do Kinect e `requirements-kinect.txt`.

Para conferir a binding antes de conectar a câmera, execute `python -c "import freenect; print('freenect importado')"` dentro do venv. A disponibilidade do dispositivo e as permissões USB ainda precisam ser verificadas no hardware usado.

## Executar

Com o ambiente ativado:

```bash
python dance_interactive_poc.py
```

Ou sem ativá-lo:

```bash
.venv/bin/python dance_interactive_poc.py
```

Pressione `q` em uma janela da aplicação para encerrar. No Kinect, os atalhos alteram a faixa em passos de 25 mm:

| Tecla | Ação |
| --- | --- |
| `a` | Reduz o limite mínimo |
| `z` | Aumenta o limite mínimo |
| `s` | Reduz o limite máximo |
| `x` | Aumenta o limite máximo |

Esses ajustes só se aplicam ao Kinect. No fallback, a máscara vem da segmentação do MediaPipe; o mapa de profundidade é preenchido com zeros.

`Ctrl+C` também solicita o encerramento. A captura Kinect usa chamadas síncronas nativas; o programa envia `sync_stop()` e aguarda a thread por tempo limitado para evitar ficar preso se o driver não responder.

## Janelas e telemetria

- **Debug & Tracking:** vídeo com máscara/landmarks e FPS, latência e limites do sensor.
- **Projetor 1 - Chão/Fundo:** ondas e círculos guiados pelo centro de massa da máscara.
- **Projetor 2 - Corpo/Frontal:** padrão generativo aplicado apenas dentro da máscara binária.

A latência exibida é medida por `time.perf_counter()` desde o timestamp associado ao frame até o fim do ciclo de exibição/`waitKey`. É uma estimativa de software; não mede exposição do sensor, sincronização real dos projetores ou o tempo até o conteúdo aparecer fisicamente. Os timestamps de Kinect e webcam também não são equivalentes, portanto os resultados entre modos não devem ser comparados como uma medição calibrada.

O indicador usa verde até 35 ms, amarelo acima de 35 até 45 ms e vermelho acima de 45 ms. O FPS representa a taxa observada pelo loop visual, não necessariamente a taxa nativa do sensor.

## Roadmap

1. **Validar hardware e instalação:** testar Kinect 1414/1473, fallback de webcam e encerramento em cada backend.
2. **Calibrar latência:** documentar timestamp de captura em cada sensor, separar idade do frame de processamento/renderização e medir com ferramenta externa a resposta física do projetor.
3. **Melhorar interação:** estabilizar o centro de massa, detectar pés/partes do corpo e disparar eventos de onda por movimento.
4. **Preparar projeção:** permitir configuração de resolução, posição das janelas, fullscreen por projetor e calibração geométrica.
5. **Consolidar operação:** adicionar testes automatizados para funções sem hardware e registrar FPS/latência em execuções longas.

## Limitações conhecidas

- OpenCV pode abrir as duas janelas de projetor em tela cheia no mesmo monitor; a disposição depende do sistema de janelas e ainda não há calibração de projetores.
- A segmentação MediaPipe e os drivers Kinect não foram validados neste ambiente.
- A combinação de sistema operacional, driver, câmera, resolução e projetor determina a latência real.

### Kinect: avisos USB e áudio

`Could not open audio: -1` refere-se ao dispositivo/canal de áudio, que esta PoC não utiliza para RGB ou profundidade. Já mensagens como `USB device disappeared`, `control transfer failed` e `USB camera marked dead` indicam falha de comunicação com o Kinect. Verifique cabo e alimentação, conexão direta a uma porta USB 2.0, permissões udev e se outro processo está usando o Kinect. Se a thread nativa continuar bloqueada após o pedido de parada, ela é daemon e não impede a tentativa de fallback nem a saída do processo; porém, a libfreenect pode continuar emitindo logs até o processo encerrar.

**Partida lenta é normal neste equipamento:** o primeiro frame do Kinect chega em ~6 a 8 s depois de abrir o dispositivo (medido; depois disso a captura mantém 30 fps). A PoC espera até 20 s antes de considerar o Kinect indisponível. Um `Lost N total packets` isolado no início do stream e um `Expected 1748 data bytes` ocasional são esperados e não indicam defeito; o sinal de problema é perda contínua, `resyncing` ou `control transfer failed` repetidos.

O fallback para webcam é feito apenas na inicialização. Se nem Kinect nem webcam estiverem disponíveis, a PoC encerra com uma mensagem indicando os dois erros. Confirme a presença da câmera com `ls /dev/video*` e teste o Kinect independentemente com `freenect-glview` antes de executar a PoC.

Para um veredito objetivo sobre o Kinect, rode `python kinect_diagnostic.py [segundos]` (padrão: 15 s medidos após o primeiro frame). Ele lista os dispositivos USB com velocidade e controlador, captura profundidade/RGB com `runloop` em um subprocesso (que pode ser encerrado se travar) e informa o tempo até o primeiro frame, o FPS em regime estável, o maior intervalo entre frames e as perdas registradas pela libfreenect. Retorna `0` (OK), `1` (instável) ou `2` (falha). Se o veredito for instável, repita após trocar de porta ou cabo; o Kinect v1 às vezes é mais estável em portas EHCI (`ehci-pci` no `lsusb -t`) do que em xHCI (`xhci_hcd`), mas neste equipamento a captura foi estável em xHCI.
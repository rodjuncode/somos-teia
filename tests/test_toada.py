"""Toada do caipira: um canal por pessoa, sorteado, com volume pela excitacao das cordas.

Pessoas e cordas sinteticas (caixas e nos deslocados dentro delas) e relogio simulado:
testa a atribuicao de canais e a dinamica de volume sem camera, YOLO nem audio.

    python -m unittest discover -s tests -v
"""

import collections
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dance_interactive_poc as poc  # noqa: E402

FPS = 30.0
SIZE = (1280, 720)
CHANNELS = list(range(1, poc.VIOLA_CHANNELS + 1))
# Deslocamento das cordas, em px, de quem esta parado e de quem danca forte
# (medidos no video de referencia: 0-2 px parado, ~15 px dancando forte).
STILL_PX = 1.0
DANCE_PX = 15.0


def box_at(index: int) -> tuple[int, int, int, int]:
    """Caixa da pessoa index: lado a lado, bem separadas, como dancarinos em fila."""
    x0 = 20 + index * 180
    return (x0, 200, x0 + 140, 600)


class FakeField:
    """Malha de nos em grade: desloca os nos dentro das caixas de quem se mexe."""

    def __init__(self) -> None:
        xs, ys = np.meshgrid(np.arange(0, SIZE[0], 10.0), np.arange(0, SIZE[1], 10.0))
        self.positions = np.stack((xs.ravel(), ys.ravel()), axis=1).astype(np.float32)
        self.moves = np.zeros_like(self.positions)

    def set_people(self, people: list[tuple[tuple[int, int, int, int], float]]) -> None:
        self.moves[:] = 0.0
        for (x0, y0, x1, y1), pixels in people:
            inside = (
                (self.positions[:, 0] >= x0) & (self.positions[:, 0] <= x1)
                & (self.positions[:, 1] >= y0) & (self.positions[:, 1] <= y1)
            )
            self.moves[inside, 0] = pixels

    def node_motion(self) -> tuple[np.ndarray, np.ndarray]:
        return self.positions, self.moves


class Stage:
    """O caipira com pessoas sinteticas, quadro a quadro."""

    def __init__(self) -> None:
        self.time = 0.0
        self.processor = poc.VisionProcessor("caipira")
        self.processor.clock = lambda: self.time
        self.field = FakeField()

    def run(self, people: list[tuple[tuple[int, int, int, int], float]], seconds: float) -> None:
        for _ in range(int(round(seconds * FPS))):
            self.time += 1.0 / FPS
            self.field.set_people(people)
            self.processor._play_toada(
                [box for box, _ in people], None, np.ones(2, np.float32), SIZE, self.field
            )

    def slots(self) -> list[int]:
        return [player.slot for player in self.processor._viola_players]

    def audible(self) -> set[int]:
        return {channel for channel, volume in self.processor.channel_volumes.items() if volume > 0}

    def volume(self, channel: int) -> float:
        return self.processor.channel_volumes[channel]


class ChannelCountTest(unittest.TestCase):
    def test_each_dancer_gets_exactly_one_channel(self):
        for count in range(1, len(CHANNELS) + 1):
            with self.subTest(dancers=count):
                stage = Stage()
                stage.run([(box_at(index), DANCE_PX) for index in range(count)], 5.0)
                slots = stage.slots()
                self.assertEqual(len(slots), count)
                self.assertEqual(len(set(slots)), count, "duas pessoas no mesmo canal")
                self.assertEqual(stage.audible(), set(slots), "canal soando sem dono")

    def test_more_people_than_channels(self):
        stage = Stage()
        stage.run([(box_at(index), DANCE_PX) for index in range(7)], 3.0)
        self.assertEqual(sorted(stage.slots()), CHANNELS)
        self.assertEqual(stage.audible(), set(CHANNELS))

    def test_still_people_stay_silent(self):
        stage = Stage()
        stage.run([(box_at(index), STILL_PX) for index in range(3)], 10.0)
        self.assertEqual(len(stage.slots()), 3)
        self.assertEqual(stage.audible(), set())

    def test_only_the_dancer_sounds(self):
        stage = Stage()
        stage.run([(box_at(0), DANCE_PX), (box_at(1), STILL_PX), (box_at(2), STILL_PX)], 5.0)
        dancer = stage.processor._viola_players[0].slot
        self.assertEqual(stage.audible(), {dancer})

    def test_leaving_frees_the_channel(self):
        stage = Stage()
        stage.run([(box_at(0), DANCE_PX), (box_at(1), DANCE_PX)], 3.0)
        leaving = stage.processor._viola_players[1].slot
        stage.run([(box_at(0), DANCE_PX)], poc.CAIPIRA_TRACK_GRACE + 0.5)
        self.assertNotIn(leaving, stage.slots())
        self.assertEqual(len(stage.slots()), 1)
        # Sem dono, o canal so decai ate o silencio.
        before = stage.volume(leaving)
        stage.run([(box_at(0), DANCE_PX)], 1.0)
        self.assertLess(stage.volume(leaving), before)
        stage.run([(box_at(0), DANCE_PX)], 30.0)
        self.assertEqual(stage.volume(leaving), 0.0)
        self.assertEqual(len(stage.audible()), 1)


class RandomAssignmentTest(unittest.TestCase):
    TRIALS = 1000

    def test_first_channel_is_random(self):
        counts = collections.Counter()
        for _ in range(self.TRIALS):
            stage = Stage()
            stage.run([(box_at(0), DANCE_PX)], 1.0 / FPS)
            counts[stage.slots()[0]] += 1
        # Uniforme: 20% cada; a margem e de ~5 desvios padrao com 1000 sorteios.
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                self.assertGreater(counts[channel] / self.TRIALS, 0.13, counts)
                self.assertLess(counts[channel] / self.TRIALS, 0.27, counts)

    def test_next_channel_is_random_among_free(self):
        # Com o canal da primeira pessoa fixado, a segunda sorteia entre os 4 livres.
        counts = collections.Counter()
        for _ in range(self.TRIALS):
            stage = Stage()
            stage.run([(box_at(0), DANCE_PX)], 0.5)
            first = stage.slots()[0]
            stage.run([(box_at(0), DANCE_PX), (box_at(1), DANCE_PX)], 1.0 / FPS)
            second = stage.slots()[1]
            self.assertNotEqual(first, second)
            counts[(second - first) % len(CHANNELS)] += 1
        for offset in range(1, len(CHANNELS)):
            with self.subTest(offset=offset):
                self.assertGreater(counts[offset] / self.TRIALS, 0.17, counts)
                self.assertLess(counts[offset] / self.TRIALS, 0.33, counts)

    def test_order_differs_between_runs(self):
        orders = set()
        for _ in range(50):
            stage = Stage()
            for count in range(1, len(CHANNELS) + 1):
                stage.run([(box_at(index), DANCE_PX) for index in range(count)], 0.2)
            orders.add(tuple(stage.slots()))
        self.assertGreater(len(orders), 10, "a ordem dos canais quase nao varia")


class VolumeDynamicsTest(unittest.TestCase):
    def test_rises_gradually_up_to_the_maximum(self):
        stage = Stage()
        full = poc.CAIPIRA_EXCITE_FULL_PX
        stage.run([(box_at(0), full)], 1.0)
        channel = stage.slots()[0]
        self.assertAlmostEqual(stage.volume(channel), poc.CAIPIRA_VOLUME_RISE, delta=0.02)
        stage.run([(box_at(0), full)], 1.0 / poc.CAIPIRA_VOLUME_RISE)
        self.assertEqual(stage.volume(channel), 1.0)

    def test_weaker_interaction_rises_slower(self):
        strong, weak = Stage(), Stage()
        strong.run([(box_at(0), poc.CAIPIRA_EXCITE_FULL_PX)], 2.0)
        weak.run([(box_at(0), poc.CAIPIRA_EXCITE_FULL_PX / 2)], 2.0)
        self.assertAlmostEqual(
            weak.volume(weak.slots()[0]), strong.volume(strong.slots()[0]) / 2, delta=0.02
        )

    def test_decays_nonlinearly_without_interaction(self):
        stage = Stage()
        stage.run([(box_at(0), DANCE_PX)], 6.0)
        channel = stage.slots()[0]
        peak = stage.volume(channel)
        # Segura por CAIPIRA_VOLUME_HOLD, depois perde metade a cada meia-vida: a
        # queda em volume e maior no comeco que no fim (nao linear).
        stage.run([(box_at(0), STILL_PX)], poc.CAIPIRA_VOLUME_HOLD)
        self.assertAlmostEqual(stage.volume(channel), peak, delta=1e-6)
        half_life = poc.CAIPIRA_VOLUME_HALF_LIFE
        stage.run([(box_at(0), STILL_PX)], half_life)
        first = stage.volume(channel)
        self.assertAlmostEqual(first, peak / 2, delta=0.02)
        stage.run([(box_at(0), STILL_PX)], half_life)
        second = stage.volume(channel)
        self.assertAlmostEqual(second, peak / 4, delta=0.02)
        self.assertGreater(peak - first, first - second)


if __name__ == "__main__":
    unittest.main()

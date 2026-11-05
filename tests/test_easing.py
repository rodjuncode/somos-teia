"""Transicoes dos parametros do teclado: a mola (Smoothed) e a malha refeita (adopt).

    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dance_interactive_poc as poc  # noqa: E402

FPS = 30.0


def run(value: poc.Smoothed, target: float, seconds: float) -> list[float]:
    return [value.step(target, 1.0 / FPS) for _ in range(round(seconds * FPS))]


class SmoothedTest(unittest.TestCase):
    def test_eases_in_and_out_without_overshoot(self):
        values = run(poc.Smoothed(0.0), 1.0, 2.0)
        steps = np.diff([0.0] + values)
        self.assertTrue(np.all(steps >= 0.0), "voltou para tras")
        self.assertLessEqual(max(values), 1.0)
        # Comeca devagar, acelera e desacelera no fim.
        self.assertLess(steps[0], steps[3])
        self.assertLess(steps[-1], steps[3])
        reached = values[round(poc.PARAM_EASE_SECONDS * FPS) - 1]
        self.assertGreater(reached, 0.97)
        self.assertEqual(values[-1], 1.0, "nao grudou no alvo exato")

    def test_retarget_keeps_velocity(self):
        value = poc.Smoothed(0.0)
        before = run(value, 1.0, 0.2)
        after = run(value, 2.0, 0.2)
        # Sem tranco: o passo logo depois da troca de alvo nao cai a zero.
        self.assertGreater(after[0] - before[-1], 0.5 * (before[-1] - before[-2]))

    def test_hue_takes_the_short_way_round(self):
        value = poc.Smoothed(350.0, period=360.0, epsilon=0.05)
        values = run(value, 10.0, 1.5)
        self.assertTrue(all(v >= 349.0 or v <= 11.0 for v in values), "deu a volta longa")
        self.assertEqual(values[-1], 10.0)

    def test_log_scale_for_multiplicative_steps(self):
        # Em escala log, a metade do caminho de 1 a 4 e 2 (a media geometrica): a
        # densidade passa por 2 no mesmo quadro em que uma mola linear passa por 0,5.
        density = run(poc.Smoothed(1.0, log=True), 4.0, 2.0)
        linear = run(poc.Smoothed(0.0), 1.0, 2.0)
        self.assertEqual(
            next(i for i, v in enumerate(density) if v >= 2.0 - 1e-6),
            next(i for i, v in enumerate(linear) if v >= 0.5 - 1e-6),
        )
        self.assertEqual(density[-1], 4.0)


class AdoptTest(unittest.TestCase):
    def make(self, density: float, rotation: float) -> poc.PlaidStack:
        processor = poc.VisionProcessor("caipira")
        processor._live["density"] = poc.Smoothed(density, log=True)
        processor._live["rotation"] = poc.Smoothed(rotation, period=360.0)
        return processor, processor._make_field(1280, 720)

    def deform(self, field: poc.PlaidStack) -> None:
        rng = np.random.default_rng(1)
        for layer, _ in field._layers:
            for family in layer._families:
                family._displacement[:, 1:-1] = rng.normal(0, 8, family._displacement[:, 1:-1].shape)
            layer.pulses._spawn(10)

    def stats(self, field: poc.PlaidStack) -> tuple[float, int]:
        total, count = 0.0, 0
        for layer, _ in field._layers:
            for family in layer._families:
                total += float(np.hypot(*family._displacement.reshape(-1, 2).T).sum())
                count += family._displacement.shape[0] * family._displacement.shape[1]
        return total / count, field.pulse_count

    def test_small_rotation_keeps_deformation_and_pulses(self):
        processor, old = self.make(1.0, 45.0)
        self.deform(old)
        processor._live["rotation"] = poc.Smoothed(46.0, period=360.0)
        new = processor._make_field(1280, 720)
        new.adopt(old)
        (old_mean, old_pulses), (new_mean, new_pulses) = self.stats(old), self.stats(new)
        self.assertAlmostEqual(new_mean, old_mean, delta=0.1 * old_mean)
        self.assertGreaterEqual(new_pulses, 0.9 * old_pulses)

    def test_density_step_keeps_most_of_the_deformation(self):
        processor, old = self.make(1.0, 45.0)
        self.deform(old)
        processor._live["density"] = poc.Smoothed(1.05, log=True)
        new = processor._make_field(1280, 720)
        new.adopt(old)
        (old_mean, _), (new_mean, new_pulses) = self.stats(old), self.stats(new)
        # Com mais densidade entram linhas novas, em repouso.
        self.assertGreater(new_mean, 0.8 * old_mean)
        self.assertGreater(new_pulses, 0)

    def test_pulses_stay_on_their_lines(self):
        processor, old = self.make(1.0, 45.0)
        self.deform(old)
        processor._live["rotation"] = poc.Smoothed(45.5, period=360.0)
        new = processor._make_field(1280, 720)
        new.adopt(old)
        self.assertGreater(new.pulse_count, 0, "nenhum pulso herdado")
        for layer, _ in new._layers:
            pulses = layer.pulses
            self.assertTrue(np.all(pulses._line < pulses._line_counts[pulses._family]))
            self.assertTrue(np.all(pulses._position >= -pulses._tail))


class ProcessorEasingTest(unittest.TestCase):
    def test_keys_set_targets_and_render_follows(self):
        processor = poc.VisionProcessor("caipira")
        clock = {"t": 0.0}
        processor.clock = lambda: clock["t"]
        start = processor.live_hue
        processor.adjust_plaid(hue_delta=40.0)
        self.assertEqual(processor.plaid_hue, (start + 40.0) % 360.0, "alvo nao mudou na hora")
        seen = []
        for _ in range(round(1.0 * FPS)):
            clock["t"] += 1.0 / FPS
            processor._ease_parameters()
            seen.append(processor.live_hue)
        self.assertNotEqual(seen[0], processor.plaid_hue, "pulou direto para o alvo")
        self.assertEqual(seen[-1], processor.plaid_hue)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
from tools.evaluate_bags import evaluate


class OriginalEvidenceTests(unittest.TestCase):
    def test_original_rep2_results(self):
        expected={1:(5267,.005305587281774626,.008369137602304939),
                  2:(9347,.016095670225804437,.0305717924179318),
                  3:(6268,.024816918791894994,.03891121789400421)}
        root=Path(__file__).resolve().parents[1]
        for i,(samples,rms,peak) in expected.items():
            with self.subTest(path=i):
                data=evaluate(root/'data/original'/f'tyr_sup_frenet_path{i}_rep2_0.db3')
                self.assertEqual(data['n_odom'],samples)
                self.assertAlmostEqual(data['rms_cte_m'],rms,places=12)
                self.assertAlmostEqual(data['max_cte_m'],peak,places=12)


if __name__=='__main__':unittest.main()

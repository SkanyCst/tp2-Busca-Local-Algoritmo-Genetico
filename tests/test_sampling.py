import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.benchmark_sa import run_sampling_for_instance


class TestSamplingBenchmark(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        self.pequena_path = os.path.join(self.data_dir, "manutencao_pequena.txt")
        self.fig1_path = os.path.join(self.data_dir, "figura1.txt")

    def test_amostragem_instancia_pequena(self):
        res = run_sampling_for_instance(
            self.pequena_path,
            budget=2000,
            runs=3,
            base_seed=42,
            verbose=False,
        )
        self.assertEqual(res["runs"], 3)
        self.assertEqual(res["n"], 12)
        self.assertGreater(res["mean_cost"], 0)
        self.assertGreater(res["mean_time"], 0)
        self.assertGreater(res["mean_init_rate"], 0.60)
        self.assertLess(res["mean_fin_rate"], 0.10)
        self.assertEqual(len(res["raw_runs"]), 3)

    def test_amostragem_diferentes_tamanhos_de_entrada(self):
        for filename in ["figura1.txt", "manutencao_pequena.txt"]:
            filepath = os.path.join(self.data_dir, filename)
            res = run_sampling_for_instance(
                filepath,
                budget=1000,
                runs=2,
                base_seed=10,
                verbose=False,
            )
            self.assertTrue(res["min_cost"] <= res["median_cost"] <= res["max_cost"])
            self.assertEqual(len(res["raw_runs"]), 2)

    def test_arquivo_inexistente_lanca_erro(self):
        with self.assertRaises(FileNotFoundError):
            run_sampling_for_instance("arquivo_inexistente.txt", budget=100, runs=1)


if __name__ == "__main__":
    unittest.main()

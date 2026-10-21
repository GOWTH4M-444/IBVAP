import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))


class Evaluate(unittest.TestCase):
    def test_score_matching(self):
        spec = importlib.util.spec_from_file_location("evaluate", os.path.join(ROOT, "scripts", "evaluate.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        pred = [("A", 10.0), ("A", 10.5), ("A", 50.0), ("B", 5.0)]
        truth = [("A", 11.0), ("A", 30.0)]
        r = m.score(pred, truth, 2.0)
        self.assertEqual((r["A"]["tp"], r["A"]["fp"], r["A"]["fn"]), (1, 2, 1))   # 10.5 can't re-use the same label
        self.assertEqual((r["B"]["fp"], r["B"]["tp"]), (1, 0))
        self.assertEqual(r["A"]["precision"], 0.333)


if __name__ == "__main__":
    unittest.main()

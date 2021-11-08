import unittest
from si.unsupervised import PCA


class MyTestCase(unittest.TestCase):
    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/cpu.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)

    def test_PCA(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()

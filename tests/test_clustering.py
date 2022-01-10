import unittest
from src.si.unsupervised import clustering


class TestClustering(unittest.TestCase):
    def setUp(self):
        from src.si.data import Dataset
        self.filename = "datasets/cpu.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)

    def test_KMeans(self):
        kmeans = clustering.KMeans(10)
        kmeans.fit(self.dataset)
        a, b = kmeans.transform(self.dataset)
        self.assertEqual(a.shape, (10, 6))
        self.assertEqual(b.shape, (209,))


if __name__ == '__main__':
    unittest.main()

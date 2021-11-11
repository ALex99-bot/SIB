import unittest
from si.unsupervised import PCA


class MyTestCase(unittest.TestCase):
    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/cpu.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)

    def test_PCA(self):
        pca = PCA(2)
        self.assertEqual(pca.fit_transform(self.dataset).shape, (209,2))


if __name__ == '__main__':
    unittest.main()

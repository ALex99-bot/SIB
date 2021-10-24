import unittest
from si.data import feature_selection


class TestFRegression(unittest.TestCase):

    def setUp(self):
        from si.data import Dataset
        self.filename = "datasets/lr-example1.data"
        self.dataset = Dataset.from_data(self.filename, labeled=True)

    def test_f_regression(self):
        self.assertEqual(feature_selection.f_regression(self.dataset)[0].shape, (1,))


if __name__ == '__main__':
    unittest.main()

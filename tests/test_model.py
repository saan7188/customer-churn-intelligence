import unittest
from churn.data import sample_data, validate
from churn.model import train

class ModelTests(unittest.TestCase):
    def test_sample(self):
        self.assertEqual(len(validate(sample_data())), 600)
    def test_invalid_data(self):
        with self.assertRaises(ValueError):
            validate(sample_data().drop(columns="churn"))
    def test_logistic(self):
        _, metrics, matrix = train(sample_data())
        self.assertEqual(matrix.shape, (2,2))
        self.assertTrue(0 <= metrics["roc_auc"] <= 1)
    def test_forest(self):
        _, metrics, _ = train(sample_data(), "forest")
        self.assertTrue(0 <= metrics["f1"] <= 1)

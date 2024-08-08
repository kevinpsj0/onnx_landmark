import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import os
from models.model_loader import ModelLoader

class TestModelLoader(unittest.TestCase):
    def setUp(self):
        # Path to the ONNX model for testing
        self.model_path = "models/model.onnx"
        self.model_loader = ModelLoader(self.model_path)

    def test_model_loader(self):
        # Check if the model file exists
        self.assertTrue(os.path.exists(self.model_path), f"Model file {self.model_path} does not exist.")
        
        # Try loading the model
        try:
            model = self.model_loader.load_model()
            self.assertIsNotNone(model, "Failed to load the model.")
        except Exception as e:
            self.fail(f"ModelLoader raised an exception: {e}")



if __name__ == "__main__":
    unittest.main()
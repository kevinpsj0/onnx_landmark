import unittest
import shutil
import os 
import numpy as np
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.data_loader import DataLoader
from dataset.preprocess import Preprocessor

num_points = 8000

class TestDataLoader(unittest.TestCase):
    @classmethod
    def test_data_loader_length(self):
        data_dir = 'tests/test_data'
        file_idx = 0 

        dataloader = DataLoader(data_dir, file_idx=file_idx, pretransform=Preprocessor())

        # Check the length of cropped data
        expected_length = 24
        actual_length = len(dataloader)
        assert actual_length == expected_length, f"Expected {expected_length}, got {actual_length}"

    def test_data_loader_contents(self):
        data_dir = 'tests/test_data'
        file_idx = 0 

        dataloader = DataLoader(data_dir, file_idx=file_idx, pretransform=Preprocessor())

        x, y, id = dataloader[0]
        
        # check shape and type of data
        for x, y, id in dataloader : 
            self.assertEqual(x.shape, (num_points,3), f"Shape of x expected ({num_points}, 3), got {x.shape}")
            self.assertEqual(y.shape, (1,18), f"Shape of y (1,18), got {y.shape}")
            self.assertIsInstance(id, str, f"id needs to be str, got {type(id)}")

    def test_data_loader_pretransform(self):
        data_dir = 'tests/test_data'
        file_idx = 0 

        dataloader = DataLoader(data_dir, file_idx=file_idx, pretransform=Preprocessor())

        for x, y, id in dataloader:

            # check x normalized
            self.assertTrue(np.all((x >= 0) & (x <= 1)), "All values in x should be between 0 and 1")

            # check coordinates normalized
            self.assertTrue(np.all((y[0, :15] >= 0) & (y[0, :15] <= 1)), "First 15 values in y should be between 0 and 1")

            # check axis integrity
            direction_vector = y[0, 15:18]
            magnitude = np.linalg.norm(direction_vector)
            self.assertTrue(np.allclose(magnitude, 1, atol=1e-6), f"The magnitude of the direction vector should be 1, got {magnitude}")


if __name__ == "__main__":
    unittest.main()
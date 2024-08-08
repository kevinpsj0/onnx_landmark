import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.helpers import import_data

class DataLoader(Dataset):
    def __init__(self, data_dir, file_idx = None, file_name = None, pretransform = None):
        assert file_name is not None or file_idx is not None 
        self.data_dir = data_dir
        self.case_dir = None
        self.case_name = None
        self.raw_data = {}
        self.processed_data = []
        self.pretransform = pretransform

        self._load_case(file_name, file_idx)
        self._process()

    
    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

    def _load_case(self, file_name, file_idx) : 
        cases = [case for case in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, case))]
        self.case_name = file_name if file_name else cases[file_idx]
        self.case_dir = os.path.join(self.data_dir, self.case_name)

    def _process(self) : 
        data_upper = import_data(self.case_dir, "Upper")
        data_lower = import_data(self.case_dir, "Lower")

        if data_upper is None or data_lower is None:
            raise FileNotFoundError(f"Data files missing for case: {self.case_name}")
        

        self.raw_data['Upper'] = data_upper
        self.raw_data['Lower'] = data_lower
        self.processed_data += self.pretransform(data_upper)
        self.processed_data += self.pretransform(data_lower)

    def get_raw(self, pre) : 
        return self.raw_data[pre]
        
import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

'''
get knowledge dataset, pretty much from ROME 
'''

class KnownsDataset(Dataset):
    def __init__(self, data_dir = "./data": str, mask = True: bool, *args, **kwargs):
        data_dir = Path(data_dir)
        if(mask):
            known_loc = data_dir / "known_1000_mask.json"
        else:
            known_loc = data_dir / "known_1000_ori.json"
        if not known_loc.exists():
            print(f"{known_loc} does not exist.")

        with open(known_loc, "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

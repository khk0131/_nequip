from pathlib import Path
from typing import List, Dict, Tuple, Union
from torch.types import Device
import torch
import pickle

class NequipDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_paths: List[Path],
        device: Device,
    ):
        super().__init__()
        self.device = device
        self.dataset_frames_paths = []
        self.inputs = []
        self.outputs = []
        # self.data: List[List[Dict[str, Union[str, torch.tensor]], Dict[str, torch.tensor]]] = []
        # self.outputs: List[Dict[str, torch.tensor]] = []
        for datase_path in dataset_paths:
            dataset_frames_paths = list(datase_path.glob('**/*.pickle'))
            for dataset_frames_path in dataset_frames_paths:
                with open(dataset_frames_path, 'rb') as f:
                    frames = pickle.load(f)
                    for frame_idx in range(len(frames)):
                        input_data: Dict[str, Union[str, torch.tensor]] = {}
                        output_data: Dict[str, Union[str, torch.tensor]] = {}
                        input_data['pos'] = torch.tensor(frames[frame_idx]['pos'], device=self.device, dtype=torch.float32)
                        input_data['edge_index'] = torch.tensor(frames[frame_idx]['edge_index'], device=self.device, dtype=torch.long)
                        input_data['cell'] = torch.tensor(frames[frame_idx]['cell'], device=self.device, dtype=torch.float32)
                        output_data['potential_energy'] = torch.tensor(frames[frame_idx]['potential_energy'], device=self.device, dtype=torch.float32)
                        input_data['atom_types'] = torch.tensor(frames[frame_idx]['atom_types'], device=self.device, dtype=torch.long)
                        output_data['force'] = torch.tensor(frames[frame_idx]['force'], device=self.device, dtype=torch.float32)
                        input_data['cut_off'] = torch.tensor(frames[frame_idx]['cut_off'], device=self.device, dtype=torch.float32)
                        input_data['path'] = str(dataset_frames_path.name)[:-7]
                        self.inputs.append(input_data)
                        self.outputs.append(output_data)
                        # concat_data.append(input_data)
                        # concat_data.append(output_data)
                        # self.data.append(concat_data)
                        # self.num_data += 1
                # print(dataset_frames_path, flush=True)
                # if dataset_frames_path == Path("/nfshome18/khosono/work_vasp/vasp/dataset/train_dataset/test_limda_20240217/Fe_CH32NC_20240210/Fe_CH32NC_20240210_5.pickle"):
                #     self.dataset_frames_paths.append(dataset_frames_path)
                
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # input_data = self.data[idx][0]
        # output_data = self.data[idx][1]

        # dataset_frames_path = self.dataset_frames_paths[idx]
        # count: int = 0
        # inputs, outputs = [], []
        # for dataset_frames_path in self.dataset_frames_paths:
        #     with open(dataset_frames_path, 'rb') as f:
        #         frames = pickle.load(f)
        #     for frame_idx in range(len(frames)):
        #         data: Dict[str, Union[str, torch.Tensor]]
        #         data['pos'] = torch.tensor(frames[frame_idx]['pos'], device=self.device, dtype=torch.float32)
        #         data['edge_index'] = torch.tensor(frames[frame_idx]['edge_index'])
        
        # with open(dataset_frames_path, "rb") as f:
        #     frames = pickle.load(f)
        # for frame_idx in range(len(frames)):
        #     frames[frame_idx]['pos'] = torch.tensor(frames[frame_idx]['pos'], device=self.device, dtype=torch.float32)
        #     frames[frame_idx]['edge_index'] = torch.tensor(frames[frame_idx]['edge_index'], device=self.device, dtype=torch.long)
        #     frames[frame_idx]['cell'] = torch.tensor(frames[frame_idx]['cell'], device=self.device, dtype=torch.float32)
        #     frames[frame_idx]['potential_energy'] = torch.tensor(frames[frame_idx]['potential_energy'], device=self.device, dtype=torch.float32)
        #     frames[frame_idx]['atom_types'] = torch.tensor(frames[frame_idx]['atom_types'], device=self.device, dtype=torch.long)
        #     frames[frame_idx]['force'] = torch.tensor(frames[frame_idx]['force'], device=self.device, dtype=torch.float32)
        #     frames[frame_idx]['cut_off'] = torch.tensor(frames[frame_idx]['cut_off'], device=self.device, dtype=torch.float32)
        #     frames[frame_idx]['path'] = str(dataset_frames_path)
        return self.inputs[idx], self.outputs[idx]
            
        
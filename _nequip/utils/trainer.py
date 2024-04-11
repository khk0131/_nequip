import torch
from typing import Dict, List, Any
import pathlib 
import os
import time
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
from .dataset import NequipDataset
from ..nn.nequip import Nequip
from e3nn.util.jit import script

class Trainer:
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        self.check_and_set_config(config)
        
        self.device = torch.device(self.config['device'])
        
        if config['model'] == 'Nequip':
            self.model = Nequip(
                num_atom_types=config['num_atom_types'],
                r_max=config['cut_off'],
                lmax=config['lmax'],
                parity=config['parity'],
                num_layers=config['num_layers'],
                num_features=config['num_features'],
                activation_function=config["activation_function"],
                invariant_layers=config['invariant_layers'],
                invariant_neurons=config['invariant_neurons'],
                bessel_num_basis=config["bessel_num_basis"],
                bessel_basis_trainable=config["bessel_basis_trainable"],
                polynomial_p=config["polynomial_p"],
                edge_sh_normalization=config["edge_sh_normalization"],
                edge_sh_normalize=config["edge_sh_normalize"],
                resnet=config["resnet"],
            )
            self.model.to(self.device)
            if self.device == 'cuda':
                self.model = torch.nn.DataParallel(self.model)
                torch.backends.cudnn.benchmark = True
            self.model.train()
            self.model.float()
            print(flush=True)
            print('Nequip model', flush=True)
            print('-----------------------------------------------------------------------------------------------------', flush=True)
            print(self.model, flush=True)
            print('total_nequip_parameters', self.model.count_parameters(), flush=True)
            print('-----------------------------------------------------------------------------------------------------', flush=True)
        else:
            raise NotImplementedError()
        
        self.train_dataset = NequipDataset(
            dataset_paths=self.config['train_dataset_paths'],
            device=self.device,
        )
        
        def collate_fn(tuple_items):
            input = []
            output = []
            for input_data, output_data in tuple_items:
                input.append(input_data)
                output.append(output_data)
            return input, output
                
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=self.config['shuffle_dataset'],
            num_workers=self.config['dataloader_num_workers'],
            batch_size=self.config['batch_size'],
            collate_fn=collate_fn,
            # multiprocessing_context='spawn',
        )
        
        torch.autograd.set_detect_anomaly(True)
        # torch.backends.cudnn.benchmark = True
        
        self.step_num = 0
        if self.config['auto_resume']:
            state_dict_paths = list(self.config['state_dict_dir'].glob('nequip_*.pth'))
            if len(state_dict_paths) != 0:
                state_dict_paths.sort(key=lambda x: int(x.name[7:-4]))
                latest_state_dict_path = state_dict_paths[-1]
                self.model.load_state_dict(torch.load(latest_state_dict_path))
                self.step_num = int(latest_state_dict_path.name[7:-4])
                config['loss_pot_ratio_initial'] = config['loss_pot_ratio_initial'] * (config['loss_pot_ratio_gamma']**(self.step_num // config['loss_pot_ratio_step_size']))
                
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['adam_weight_decay'],
            betas=self.config['adam_betas'],
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config['steplr_step_size'],
            gamma=self.config['steplr_gamma'],
        )
        for _ in range(self.step_num):
            self.scheduler.step()
            
        self.loss_pot_ratio = config['loss_pot_ratio_initial']
        self.loss_force_ratio = config['loss_force_ratio_initial']
        
        os.makedirs(config['state_dict_dir'], exist_ok=True)
        os.makedirs(config['save_frozen_model_dir'], exist_ok=True)
             
        
    def check_and_set_config(self, config: Dict[str, Any]):
        
        if 'device' not in config:
            config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            
        assert 'train_dataset_paths' in config
        for i in range(len(config['train_dataset_paths'])):
            config['train_dataset_paths'][i] = pathlib.Path(config['train_dataset_paths'][i])
        
        if 'shuffle_dataset' not in config:
            config['shuffle_dataset'] = True
        
        if 'dataloader_num_workers' not in config:
            config['dataloader_num_workers'] = 0
        
        if 'batch_size' not in config:
            config['batch_size'] = 1
        
        if 'epoch' not in config:
            config['epoch'] = 100
            
        if 'frame_skip_num' not in config:
            config['frame_skip_num'] = 10
            
        if 'loss_pot_ratio_step_size' not in config:
            config['loss_pot_ration_step_size'] = 10000000
            
        if 'loss_pot_ratio_initial' not in config:
            config['loss_pot_ratio_initial'] = 0.01

        if 'loss_pot_ratio_gamma' not in config:
            config['loss_pot_ratio_gamma'] = 0.1
        
        if 'auto_resume' not in config:
            config['auto_resume'] = False

        # optimizer
        if 'lr' not in config:
            config['lr'] = 1e-2
        
        if 'adam_betas' not in config:
            config['adam_betas'] = (0.9, 0.999)
            
        if 'adam_weight_decay' not in config:
            config['adam_weight_decay'] = 0.0
        
        # scheduler
        if 'steplr_step_size' not in config:
            config['steplr_step_size'] = 10000000
        
        if 'steplr_gamma' not in config:
            config['steplr_step_size'] = 0.1
        
        # output
        if 'save_model_step_size' not in config:
            config['save_model_step_size'] = 10000

        if 'state_dict_dir' not in config:
            config['state_dict_dir'] = pathlib.Path('state_dicts')
        else:
            config['state_dict_dir'] = pathlib.Path(config['state_dict_dir'])


        if 'save_frozen_model_dir' not in config:
            config['save_frozen_model_dir'] = pathlib.Path('frozen_models')
        else:
            config['save_frozen_model_dir'] = pathlib.Path(config['save_frozen_model_dir'])
            
        assert 'model' in config and config['model'] == 'Nequip'
        assert 'num_atom_types' in config
        assert 'cut_off' in config
        assert 'lmax' in config
        assert 'parity' in config
        assert 'num_layers' in config
        assert 'num_features' in config
        assert 'invariant_layers' in config
        assert 'invariant_neurons' in config
        assert 'activation_function' in config
        
        self.config = config
    
    def save_model(self):
        torch.save(
            self.model.state_dict(),
            self.config['state_dict_dir'] / f'nequip_{self.step_num}.pth'
        )
        script_model = script(self.model)
        frozen_model = torch.jit.freeze(script_model.eval())
        frozen_model.save(
            self.config['save_frozen_model_dir'] / f'nequip_frozen_{self.step_num}.pth'
            )
    
    def frames_to_concated_tensor(self, input_data: List[Dict[str, torch.Tensor]]):
        num_datas = len(input_data)
        assert num_datas > 0
        true_total_energy_list = []
        true_force_list = []
        for data_idx in range(len(input_data)):
            true_total_energy_list.append(input_data[data_idx]['potential_energy'].reshape(1))
            true_force_list.append(input_data[data_idx]['force'])
        true_total_energy = torch.cat(true_total_energy_list, dim=0)
        true_force = torch.cat(true_force_list, dim=0)

        
        return true_total_energy, true_force
    
    def average_atom_num_per_data(self, input_data: List[Dict[str, torch.Tensor]]):
        total_atom_num = 0
        for data_idx in range(len(input_data)):
            total_atom_num += input_data[data_idx]['pos'].numel()
        avg_atom_num  = total_atom_num / len(input_data)
        
        return avg_atom_num
    
    def train(self):
        """Nequipモデルを訓練するクラス
           mini batch学習ができるようにした
        """
        print("epoch step_num    pred_pot_e    true_pot_e    loss_pot    loss_force      lr          loss_pot_ratio    time_step    data_path", flush=True)
        for epoch in range(self.config['epoch']):
            start_time = time.time()
            for input_data, output_data in self.train_dataloader:
                if self.step_num % self.config['loss_pot_ratio_step_size'] == 0 and self.step_num != 0:
                    self.loss_pot_ratio *= self.config['loss_pot_ratio_gamma']
                if self.step_num % self.config['loss_force_ratio_step_size'] == 0 and self.step_num != 0:
                    self.loss_force_ratio *= self.config['loss_force_ratio_gamma']
                if self.step_num % self.config['save_model_step_size'] == 0:
                    self.save_model()
                try:
                    self.step_num += 1
                    self.optimizer.zero_grad()
                    assert abs(self.config['cut_off'] - input_data[0]['cut_off'].item()) < 1e-6, 'datasetのcut_offとtrain用のconfigのcut_offが違います'
                    outputs = self.model(
                        input_data,
                    )            
                    # print(outputs)        
                    true_total_energy, true_force = self.frames_to_concated_tensor(output_data)
                    loss_pot = torch.sum(abs(outputs['total_energy'] - true_total_energy))
                    loss_pot_per_data = loss_pot / len(input_data)
                    loss_forces = torch.sum(torch.pow((outputs['atomic_force'] - true_force), 2)).sqrt()
                    loss_force_per_data = (loss_forces / len(input_data))
                    avg_data_num = self.average_atom_num_per_data(input_data)
                    loss = (loss_pot_per_data / (avg_data_num / 3.0))**(2) * self.loss_pot_ratio + (loss_force_per_data**(2) / avg_data_num) * self.loss_force_ratio
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    loss_forces_per_atom = loss_force_per_data.item() / (avg_data_num**(1/2))
                    end_time = time.time()
                    print(f"{epoch:>3} {self.step_num:>10} {torch.sum(outputs['total_energy']).item():>12.4f} {torch.sum(true_total_energy).item():>12.4f} {loss_pot_per_data.item():>12.4f}    {loss_forces_per_atom:>12.5f} {self.scheduler.get_last_lr()[0]:>12.2e} {self.loss_pot_ratio:>12.2e}    {(end_time - start_time):>12.4f}", flush=True)
                except Exception as e:
                    print(e)
                    continue

                # self.scheduler.step()
                # epoch_time = time.time()
                # print(epoch_time - start_time, flush=True)
        # epoch_end_time = time.time()
        # print(epoch_end_time - start_time, flush=True)
                        
        
        
        # for epoch in range(self.config['epoch']):
        #     for initial_frame_idx in range(self.config['frame_skip_num']):
        #         for frames in self.train_dataloader:
        #             for frame_idx in range(initial_frame_idx, len(frames), self.config['frame_skip_num']):
        #                 start_time = time.time()
        #                 if self.step_num % self.config['loss_pot_ratio_step_size'] == 0 and self.step_num != 0:
        #                     self.loss_pot_ratio *= self.config['loss_pot_ratio_gamma']
        #                 if self.step_num % self.config['loss_force_ratio_step_size'] == 0 and self.step_num != 0:
        #                     self.loss_force_ratio *= self.config['loss_force_ratio_gamma']
        #                 if self.step_num % self.config['save_model_step_size'] == 0:
        #                     self.save_model()
        #                 try:
        #                     data = frames[frame_idx]
        #                     print(data)
        #                     print(data['cut_off'])
        #                     self.step_num += 1
        #                     self.optimizer.zero_grad()
        #                     assert abs(self.config['cut_off'] - data['cut_off'].item()) < 1e-6, 'datasetのcut_offとtrain用のconfigのcut_offが違います'
        #                     # print(data, flush=True)
        #                     outputs = self.model(
        #                         data['pos'],
        #                         data['atom_types'],
        #                         data['edge_index'],
        #                         data['cut_off'],
        #                         data['cell'],
        #                     )
        #                     loss_pot = abs(outputs['total_energy'] - data['potential_energy'])
        #                     loss_forces = torch.sum((outputs['atomic_force'] - data['force']).pow(2)).sqrt()
        #                     loss = (loss_pot / (data['pos'].numel() / 3.0)).pow(2) * self.loss_pot_ratio + (loss_forces.pow(2) / data['pos'].numel()) * self.loss_force_ratio
        #                     loss.backward()
        #                     self.optimizer.step()
        #                     self.scheduler.step()
        #                     loss_forces_per_atom = loss_forces.item() / (data['pos'].numel()**(1/2))
        #                     data_path = pathlib.Path(data['path']).stem
        #                     end_time = time.time()
        #                     print(f"{epoch:>3} {self.step_num:>10} {outputs['total_energy'].item():>12.4f} {data['potential_energy'].item():>12.4f} {loss_pot.item():>12.4f}    {loss_forces_per_atom:>12.5f} {self.scheduler.get_last_lr()[0]:>12.2e} {self.loss_pot_ratio:>12.2e}    {(end_time - start_time):>12.4f}      {data_path}", flush=True)
        #                 except Exception as e:
        #                     print(e)
        #                     continue
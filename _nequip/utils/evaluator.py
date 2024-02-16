import torch
from typing import Dict, Any, List
import pathlib
import pprint
import numpy as np
import os
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from ..nn.nequip_batch import NequipBatch
from ..utils.dataset import NequipDataset


class Evaluator:
    '''Allegroモデルを評価するクラス
    '''
    def __init__(
        self,
        config: Dict[str, Any], 
    ):
        
        self.check_and_set_config(config)

        self.device = torch.device(self.config['device'])

        if 'model_step_num' in self.config:
            state_dict_paths = [self.config['state_dict_dir'] / f"nequip_{self.config['model_step_num']}.pth"]
        else:
            state_dict_paths = list(self.config['state_dict_dir'].glob('nequip_*.pth'))
            assert len(state_dict_paths) > 0
            state_dict_paths.sort(key=lambda x: int(x.name[7:-4]))

        self.state_dict_paths = state_dict_paths[::self.config['model_skip_num']]

        self.models = {}
        for state_dict_path in self.state_dict_paths:
            step_num = int(state_dict_path.name[7:-4])
            if config['model'] == 'Nequip':
                if config['activation_function'] == 'silu':
                    self.activation_function = torch.nn.functional.silu
                else:
                    raise NotImplementedError()
                
                model = NequipBatch(
                    num_atom_types=config['num_atom_types'],
                    r_max=config['cut_off'],
                    lmax=config['lmax'],
                    num_layers=config['num_layers'],
                )
                model.to(self.device)
                model.train()
                model.float()
            else:
                raise NotImplementedError()
            model.load_state_dict(torch.load(state_dict_path))
            self.models[step_num] = model

        if self.config['eval_train_dataset']:
            self.train_dataloaders = {}
            for train_dataset_path in self.config['train_dataset_paths']:
                train_dataset = NequipDataset(
                    dataset_paths=[train_dataset_path], 
                    device=self.device
                    )
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, 
                    shuffle=False,
                    num_workers=self.config['dataloader_num_workers'],
                    batch_size=None
                )
                self.train_dataloaders[train_dataset_path.name] = train_dataloader

        if self.config['eval_test_dataset']:
            self.test_dataloaders = {}
            for test_dataset_path in self.config['test_dataset_paths']:
                test_dataset = NequipDataset(
                    dataset_paths=[test_dataset_path], 
                    device=self.device
                    )
                test_dataloader = torch.utils.data.DataLoader(
                    test_dataset, 
                    shuffle=False,
                    num_workers=self.config['dataloader_num_workers'],
                    batch_size=None
                )
                self.test_dataloaders[test_dataset_path.name] = test_dataloader

        
        torch.autograd.set_detect_anomaly(True)

        os.makedirs(config['dataframe_dir'], exist_ok=True)
        
        
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
        assert 'num_layers' in config
        
        self.config = config
        
        
    def eval_dataset(self, dataloaders):
        """Nequipモデルを評価する
        """
        
        # loss_force_result['dataset_name'][step_num]
        loss_force_result = {}
        loss_pot_result = {}
        for dataset_path in dataloaders:
            loss_force_result[dataset_path] = {}
            loss_pot_result[dataset_path] = {}
            for step_num in self.models:
                loss_force_result[dataset_path][step_num] = np.nan
                loss_pot_result[dataset_path][step_num] = np.nan

        for step_num, model in self.models.items():
            for dataset_path, dataloader in dataloaders.items():
                total_loss_force = 0.0
                total_loss_pot = 0.0
                total_num_frames = 0
                for frames_all in dataloader:
                    if self.config['eval_batch_size'] == -1:
                        batch_size = len(frames_all)
                    else:
                        batch_size = self.config['eval_batch_size']
                    for frame_idx in range(0, len(frames_all), batch_size):
                        frames = frames_all[frame_idx:frame_idx+batch_size]
                        num_frames = len(frames)
                        num_atoms = frames[0]['pos'].shape[0]
                        assert len(frames) > 0
                        assert abs(self.config['cut_off'] - frames[0]['cut_off'].item()) < 1e-6, 'datasetのcut_offとconfigのcut_offが違います'
                        outputs = model(
                            frames
                        )
                        true_total_energy, true_force = self.frames_to_concated_potential_and_force(frames)
                        loss_pot = torch.sum(torch.abs(outputs['total_energy'] - true_total_energy)).item()
                        loss_pot_per_frame = loss_pot / num_frames
                        loss_force = torch.sum(torch.pow(outputs['force'] - true_force, 2)).item()
                        loss_force_per_atom_one_direction = (loss_force / (num_frames*num_atoms*3))**(1/2)
                        print(f"{step_num:>10}  {loss_pot_per_frame:>12.4f} {loss_force_per_atom_one_direction:>12.4f} {dataset_path:>30}", flush=True)
                        total_loss_pot += loss_pot
                        total_loss_force += loss_force
                        total_num_frames += num_frames
                avg_loss_force_per_atom_in_dataset = (total_loss_force / (total_num_frames*num_atoms*3))**(1/2)
                avg_loss_pot_in_dataset = total_loss_pot / total_num_frames
                loss_force_result[dataset_path][step_num] = avg_loss_force_per_atom_in_dataset
                loss_pot_result[dataset_path][step_num] = avg_loss_pot_in_dataset
        
        df_loss_pot = pd.DataFrame(loss_pot_result)
        df_loss_pot.index.name = 'step_num'
        df_loss_force = pd.DataFrame(loss_force_result)
        df_loss_force.index.name = 'step_num'


        return df_loss_pot, df_loss_force
    
    
    def frames_to_concated_potential_and_force(self, frames: List[Dict[str, torch.Tensor]]):
        """framesのpotential_energyとforceをまとめてconcatする関数
        Parameters
        ----------
            frames: List[Dict[str, torch.Tensor]]
                第一原理MDをした結果
        Returns
        -------
            true_total_energy: torch.Tensor, shape:[num_frames]
                それぞれのフレームのポテンシャルエネルギー
            true_force: torch.Tensor, shape:[num_frames, num_atoms, 3]
                それぞれのフレームの力
        """
        num_frames = len(frames)
        assert len(frames) > 0
        num_atoms = frames[0]['force'].shape[0]
        true_total_energy_list = []
        true_force_list = []
        for frame_idx in range(len(frames)):
            true_total_energy_list.append(frames[frame_idx]['potential_energy'].reshape(1))
            true_force_list.append(frames[frame_idx]['force'])
        true_total_energy = torch.cat(true_total_energy_list, dim=0)
        true_force = torch.cat(true_force_list, dim=0).reshape(num_frames, num_atoms, 3)

        return true_total_energy, true_force


    def eval(self):
        if self.config['eval_train_dataset']:
            df_loss_pot_train, df_loss_force_train = self.eval_dataset(self.train_dataloaders)
            df_loss_force_train.to_csv(self.config['dataframe_dir'] / 'loss_force_train', sep='\t')
            df_loss_pot_train.to_csv(self.config['dataframe_dir'] / 'loss_pot_train', sep='\t')

        if self.config['eval_test_dataset']:
            df_loss_pot_test, df_loss_force_test = self.eval_dataset(self.test_dataloaders)
            df_loss_force_test.to_csv(self.config['dataframe_dir'] / 'loss_force_test', sep=' ')
            df_loss_pot_test.to_csv(self.config['dataframe_dir'] / 'loss_pot_test', sep=' ')
        
        if self.config['eval_train_dataset']:
            print('df_loss_pot_train', flush=True)
            print(df_loss_pot_train, flush=True)
            print()
            print('df_loss_force_train', flush=True)
            print(df_loss_force_train, flush=True)

        if self.config['eval_test_dataset']:
            print('df_loss_pot_test', flush=True)
            print(df_loss_pot_test, flush=True)
            print()
            print('df_loss_force_test', flush=True)
            print(df_loss_force_test, flush=True)

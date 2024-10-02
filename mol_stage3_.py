import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
# from data_provider.stage3_dm import Stage3DM
from model.unimol import SimpleUniMolModel
# from model.blip2_stage3 import Blip2Stage3 
from model.mol_blip2_stage3_ import Blip2Stage3 #########
from model.dist_funs import MyDeepSpeedStrategy   
from model.llama_flash_attention import replace_llama_attn_with_flash_attn

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
# from data_provider.balance_dataset import BalanceDataset, UniformDataset
from data_provider.balance_dataset_ import BalanceDataset, UniformDataset ############
from data_provider.unimol_dataset import D3Collater

from unicore.data import Dictionary, data_utils


os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A100 gpus
torch.set_float32_matmul_precision('medium')


class Stage3DM(LightningDataModule):
    def __init__(
            self,
            mode: str = 'train',
            num_workers: int = 0,
            batch_size: int = 256,
            root: str = 'data/',
            text_max_len: int = 128,
            pad_to_multiple: int = 8,
            dictionary=None,
            tokenizer=None,
            args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.dictionary = dictionary
        self.pad_to_multiple = pad_to_multiple

        root = '/data2/project/kjh/mv-clam/' + root
        # if mode.find('pretrain') >= 0:
        if mode.find('train') >= 0:
            # self.train_dataset = BalanceDataset(root, 'pretrain', unimol_dict=dictionary, max_atoms=args.unimol_max_atoms)
            self.train_dataset = BalanceDataset(root, 'train', unimol_dict=dictionary, max_atoms=args.unimol_max_atoms)
            self.train_dataset.tokenizer = tokenizer
            self.val_dataset = UniformDataset(root, 'valid', unimol_dict=dictionary, max_atoms=args.unimol_max_atoms)
            self.val_dataset.tokenizer = tokenizer
            self.test_dataset = UniformDataset(root, 'test', unimol_dict=dictionary, max_atoms=args.unimol_max_atoms)
            self.test_dataset.tokenizer = tokenizer
        elif mode.find('eval') >= 0:
            self.test_dataset = UniformDataset(root, 'test', unimol_dict=dictionary, max_atoms=args.unimol_max_atoms)
            self.test_dataset.tokenizer = tokenizer
        else:
            raise NotImplementedError

        self.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.dictionary.pad(), self.pad_to_multiple),
        )
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.dictionary.pad(), self.pad_to_multiple),
        )
        return val_loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.dictionary.pad(), self.pad_to_multiple),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--inference_batch_size', type=int, default=8)
        parser.add_argument('--root', type=str, default='data/3d-mol-dataset')
        parser.add_argument('--text_max_len', type=int, default=384)
        return parent_parser

class TrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_token_id, pad_idx, pad_to_multiple):
        
        self.pad_idx = pad_idx
        self.tokenizer = tokenizer
        self.mol_token_id = mol_token_id
        self.pad_to_multiple = pad_to_multiple
        self.text_max_len = text_max_len
        
        # self.d3_collater = D3Collater(pad_idx)
        
    def collate_tokens_coords(
            self,
            values,
            pad_idx,
            left_pad=False,
            pad_to_length=None,
            pad_to_multiple=1,
        ):
            
            """Convert a list of 1d tensors into a padded 2d tensor."""
            size = max(v.size(0) for v in values)
            size = size if pad_to_length is None else max(size, pad_to_length)
            if pad_to_multiple != 1 and size % pad_to_multiple != 0:
                size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
            res = values[0].new(len(values), size, 3).fill_(pad_idx)
        
            def copy_tensor(src, dst):
                assert dst.numel() == src.numel()
                dst.copy_(src)
        
            for i, v in enumerate(values):
                copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
            return res    
            

    def __call__(self, batch):
        # graphs, input, output, task_type = zip(*batch)
        graphs, input, output, task_type, d2_batch = zip(*batch) ##### d2_batch 추가 input

        atom_vec, coordinates, edge_type, dist, smiles = zip(*graphs)
        padded_atom_vec = data_utils.collate_tokens(atom_vec, self.pad_idx, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms]
        padded_coordinates = self.collate_tokens_coords(coordinates, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, 3]
        padded_edge_type = data_utils.collate_tokens_2d(edge_type, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        padded_dist = data_utils.collate_tokens_2d(dist, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        # padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(graphs)
        
        graph_batch = (padded_atom_vec, padded_dist, padded_edge_type)

        input_pair = [[p, t] for p, t in zip(input, output)]


        self.tokenizer.padding_side = 'left'
        text_batch = self.tokenizer(input_pair,
                                    # truncation=True,
                                    truncation='only_second',
                                    # padding='longest',
                                    padding='max_length',
                                    add_special_tokens=True,
                                    max_length=self.text_max_len,
                                    return_tensors='pt',
                                    return_attention_mask=True,
                                    return_token_type_ids=True)

        is_mol_token = (text_batch.input_ids == self.mol_token_id)


        # assert torch.sum(is_mol_token).item() == 8*len(batch), print(input_pair)
        assert torch.sum(is_mol_token).item() == 24*len(batch), print(input_pair)

        text_batch['is_mol_token'] = is_mol_token

        d2_batch = self.d2_graph_encoder_batch(*d2_batch) ###########

        return graph_batch, text_batch, d2_batch


    def d2_graph_encoder_batch(self, *data_objects):
        # Extract node features, edge indices, and node distances
        node_features = [data.x for data in data_objects]
        adj_matrices = [data.edge_index for data in data_objects]
        node_dists = [data.node_dist for data in data_objects] ######################################################
        

        # Determine the maximum number of nodes
        max_num_nodes = max([nf.size(0) for nf in node_features])
        num_features = node_features[0].size(1)

        # Pad node features
        padded_node_features = []
        for nf in node_features:
            padding = torch.zeros((max_num_nodes - nf.size(0), num_features))
            padded_node_features.append(torch.cat([nf, padding], dim=0))
        node_feature_tensor = torch.stack(padded_node_features)

        # Pad adjacency matrices
        padded_adj_matrices = []
        for adj in adj_matrices:
            pad_size = max_num_nodes - adj.size(0)
            padded_adj = torch.nn.functional.pad(adj, (0, pad_size, 0, pad_size))
            padded_adj_matrices.append(padded_adj)
        adj_matrix_tensor = torch.stack(padded_adj_matrices)

        # Pad node distance matrices
        padded_node_dists = []
        for nd in node_dists:
            pad_size = max_num_nodes - nd.size(0)
            padded_nd = torch.nn.functional.pad(nd, (0, pad_size, 0, pad_size))
            padded_node_dists.append(padded_nd)
        node_dist_tensor = torch.stack(padded_node_dists)

        return node_feature_tensor.to(torch.float32), adj_matrix_tensor.to(torch.float32), node_dist_tensor.to(torch.float32)



class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_token_id, pad_idx, pad_to_multiple):

        self.pad_idx = pad_idx
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.mol_token_id = mol_token_id
        self.pad_to_multiple = pad_to_multiple

        # self.d3_collater = D3Collater(pad_idx)
        # self.d2_collater = Collater([], [])

    def __call__(self, batch):
        graphs, input, output, task_type, d2_batch = zip(*batch)

        atom_vec, coordinates, edge_type, dist, smiles = zip(*graphs)
        padded_atom_vec = data_utils.collate_tokens(atom_vec, self.pad_idx, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms]
        #padded_coordinates = self.collate_tokens_coords(coordinates, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, 3]
        padded_edge_type = data_utils.collate_tokens_2d(edge_type, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        padded_dist = data_utils.collate_tokens_2d(dist, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        # padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(graphs)
        
        graph_batch = (padded_atom_vec, padded_dist, padded_edge_type)

        self.tokenizer.padding_side = 'right'
        
        prompt_batch = self.tokenizer(input,
                                      truncation=False,
                                      padding='longest',
                                      add_special_tokens=True,
                                      return_tensors='pt',
                                      return_attention_mask=True,
                                      return_token_type_ids=True)
                                      # return_token_type_ids=False)

        is_mol_token = prompt_batch.input_ids == self.mol_token_id
        prompt_batch['is_mol_token'] = is_mol_token

        # assert torch.sum(is_mol_token).item() == 8 * len(batch)
        # assert torch.sum(is_mol_token).item() == 12 * len(batch)
        assert torch.sum(is_mol_token).item() == 24 * len(batch)

        target_dict = {'targets': output, 'task_type': task_type}

        d2_batch = self.d2_graph_encoder_batch(*d2_batch) ###########

        # return graph_batch, prompt_batch, target_dict
        return graph_batch, prompt_batch, d2_batch, target_dict

    def d2_graph_encoder_batch(self, *data_objects):
        # Extract node features, edge indices, and node distances
        node_features = [data.x for data in data_objects]
        adj_matrices = [data.edge_index for data in data_objects]
        node_dists = [data.node_dist for data in data_objects] ######################################################
        
        # Determine the maximum number of nodes
        max_num_nodes = max([nf.size(0) for nf in node_features])
        num_features = node_features[0].size(1)

        # Pad node features
        padded_node_features = []
        for nf in node_features:
            padding = torch.zeros((max_num_nodes - nf.size(0), num_features))
            padded_node_features.append(torch.cat([nf, padding], dim=0))
        node_feature_tensor = torch.stack(padded_node_features)

        # Pad adjacency matrices
        padded_adj_matrices = []
        for adj in adj_matrices:
            pad_size = max_num_nodes - adj.size(0)
            padded_adj = torch.nn.functional.pad(adj, (0, pad_size, 0, pad_size))
            padded_adj_matrices.append(padded_adj)
        adj_matrix_tensor = torch.stack(padded_adj_matrices)

        # Pad node distance matrices
        padded_node_dists = []
        for nd in node_dists:
            pad_size = max_num_nodes - nd.size(0)
            padded_nd = torch.nn.functional.pad(nd, (0, pad_size, 0, pad_size))
            padded_node_dists.append(padded_nd)
        node_dist_tensor = torch.stack(padded_node_dists)

        return node_feature_tensor.to(torch.float32), adj_matrix_tensor.to(torch.float32), node_dist_tensor.to(torch.float32)

# '''collater'''
# class TrainCollater:
#     def __init__(self, tokenizer, text_max_len, mol_token_id, pad_idx):
#         self.text_max_len = text_max_len
#         self.tokenizer = tokenizer
#         self.d3_collater = D3Collater(pad_idx)
#         self.mol_token_id = mol_token_id
#         self.pad_idx = pad_idx

#     def __call__(self, batch):
#         # graphs, input, output, task_type = zip(*batch)
#         graphs, input, output, task_type, d2_batch = zip(*batch) ##### d2_batch 추가 input
#         padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(graphs)
#         graph_batch = (padded_atom_vec, padded_dist, padded_edge_type)

#         input_pair = [[p, t] for p, t in zip(input, output)]

#         self.tokenizer.padding_side = 'left'
#         text_batch = self.tokenizer(input_pair,
#                                     truncation=True,
#                                     padding='longest',
#                                     add_special_tokens=True,
#                                     max_length=self.text_max_len,
#                                     return_tensors='pt',
#                                     return_attention_mask=True,
#                                     return_token_type_ids=True)

#         is_mol_token = text_batch.input_ids == self.mol_token_id

#         # assert torch.sum(is_mol_token).item() == 8*len(batch), print(input_pair)
#         assert torch.sum(is_mol_token).item() == 12*len(batch), print(input_pair)

#         text_batch['is_mol_token'] = is_mol_token

#         d2_batch = self.d2_graph_encoder_batch(*d2_batch) ###########

#         return graph_batch, text_batch, d2_batch


#     def d2_graph_encoder_batch(self, *data_objects):
#         # Extract node features, edge indices, and node distances
#         node_features = [data.x for data in data_objects]
#         adj_matrices = [data.edge_index for data in data_objects]
#         node_dists = [data.node_dist for data in data_objects] ######################################################
        

#         # Determine the maximum number of nodes
#         max_num_nodes = max([nf.size(0) for nf in node_features])
#         num_features = node_features[0].size(1)

#         # Pad node features
#         padded_node_features = []
#         for nf in node_features:
#             padding = torch.zeros((max_num_nodes - nf.size(0), num_features))
#             padded_node_features.append(torch.cat([nf, padding], dim=0))
#         node_feature_tensor = torch.stack(padded_node_features)

#         # Pad adjacency matrices
#         padded_adj_matrices = []
#         for adj in adj_matrices:
#             pad_size = max_num_nodes - adj.size(0)
#             padded_adj = torch.nn.functional.pad(adj, (0, pad_size, 0, pad_size))
#             padded_adj_matrices.append(padded_adj)
#         adj_matrix_tensor = torch.stack(padded_adj_matrices)

#         # Pad node distance matrices
#         padded_node_dists = []
#         for nd in node_dists:
#             pad_size = max_num_nodes - nd.size(0)
#             padded_nd = torch.nn.functional.pad(nd, (0, pad_size, 0, pad_size))
#             padded_node_dists.append(padded_nd)
#         node_dist_tensor = torch.stack(padded_node_dists)

#         return node_feature_tensor.to(torch.float32), adj_matrix_tensor.to(torch.float32), node_dist_tensor.to(torch.float32)



# class InferenceCollater:
#     def __init__(self, tokenizer, text_max_len, mol_token_id, pad_idx):
#         self.text_max_len = text_max_len
#         self.tokenizer = tokenizer
#         self.d3_collater = D3Collater(pad_idx)
#         self.d2_collater = Collater([], [])
#         self.mol_token_id = mol_token_id
#         self.pad_idx = pad_idx

#     def __call__(self, batch):
#         graphs, input, output, task_type, d2_batch = zip(*batch)
#         padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(graphs)
#         graph_batch = (padded_atom_vec, padded_dist, padded_edge_type)

#         self.tokenizer.padding_side = 'right'
#         prompt_batch = self.tokenizer(input,
#                                       truncation=False,
#                                       padding='longest',
#                                       add_special_tokens=True,
#                                       return_tensors='pt',
#                                       return_attention_mask=True,
#                                       return_token_type_ids=False)

#         is_mol_token = prompt_batch.input_ids == self.mol_token_id
#         prompt_batch['is_mol_token'] = is_mol_token

#         # assert torch.sum(is_mol_token).item() == 8 * len(batch)
#         assert torch.sum(is_mol_token).item() == 12 * len(batch)

#         target_dict = {'targets': output, 'task_type': task_type}

#         d2_batch = self.d2_graph_encoder_batch(*d2_batch) ###########

#         # return graph_batch, prompt_batch, target_dict
#         return graph_batch, prompt_batch, d2_batch, target_dict

#     def d2_graph_encoder_batch(self, *data_objects):
#         # Extract node features, edge indices, and node distances
#         node_features = [data.x for data in data_objects]
#         adj_matrices = [data.edge_index for data in data_objects]
#         node_dists = [data.node_dist for data in data_objects] ######################################################
        
#         # Determine the maximum number of nodes
#         max_num_nodes = max([nf.size(0) for nf in node_features])
#         num_features = node_features[0].size(1)

#         # Pad node features
#         padded_node_features = []
#         for nf in node_features:
#             padding = torch.zeros((max_num_nodes - nf.size(0), num_features))
#             padded_node_features.append(torch.cat([nf, padding], dim=0))
#         node_feature_tensor = torch.stack(padded_node_features)

#         # Pad adjacency matrices
#         padded_adj_matrices = []
#         for adj in adj_matrices:
#             pad_size = max_num_nodes - adj.size(0)
#             padded_adj = torch.nn.functional.pad(adj, (0, pad_size, 0, pad_size))
#             padded_adj_matrices.append(padded_adj)
#         adj_matrix_tensor = torch.stack(padded_adj_matrices)

#         # Pad node distance matrices
#         padded_node_dists = []
#         for nd in node_dists:
#             pad_size = max_num_nodes - nd.size(0)
#             padded_nd = torch.nn.functional.pad(nd, (0, pad_size, 0, pad_size))
#             padded_node_dists.append(padded_nd)
#         node_dist_tensor = torch.stack(padded_node_dists)

#         return node_feature_tensor.to(torch.float32), adj_matrix_tensor.to(torch.float32), node_dist_tensor.to(torch.float32)
        

def main(args):
    pl.seed_everything(args.seed)
    # model
    if args.init_checkpoint:
        model = Blip2Stage3.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage3_path:
        model = Blip2Stage3(args)
        ckpt = torch.load(args.stage3_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage3 model from {args.stage3_path}")
    elif args.stage2_path:
        model = Blip2Stage3(args)
        ckpt = torch.load(args.stage2_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    else:
        model = Blip2Stage3(args)
    print(' total params:', sum(p.numel() for p in model.parameters()))

    tokenizer = model.blip2opt.llm_tokenizer
    dm = Stage3DM(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len, 8, model.blip2opt.dictionary, tokenizer, args)

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath=f'all_checkpoints/{args.filename}/',
                                         filename='{step}',
                                         every_n_train_steps=args.every_n_train_steps,
                                         # filename='{epoch:02d}',
                                         #every_n_epochs=1, ################# SAVE EVERY EPOCH
                                         # every_n_epochs=args.save_every_n_epochs, #################
                                         save_last=True, 
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))


    if len(args.devices) > 1:
        if args.strategy_name == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = strategies.DDPStrategy(start_method='spawn')
    else:
        strategy = 'auto'
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')

    # trainer = Trainer(
    #     accelerator=args.accelerator,
    #     devices=args.devices,
    #     precision=args.precision,
    #     # max_epochs=args.max_epochs,
    #     max_steps=args.max_steps,
    #     accumulate_grad_batches=args.accumulate_grad_batches,
    #     val_check_interval=args.every_n_train_steps * args.accumulate_grad_batches,
    #     check_val_every_n_epoch=None,
    #     callbacks=callbacks,
    #     strategy=strategy,
    #     logger=logger,
    # )

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        # max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_val_batches=0, #### NO VALIDATION STEP
        num_sanity_val_steps=0, #### NO VALIDATION STEP
        # val_check_interval=args.every_n_train_steps * args.accumulate_grad_batches,
        # check_val_every_n_epoch=None,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
    )



    if args.mode.find('train') >= 0:
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    elif args.mode.find('eval') >= 0:
        trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage3_test")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # MM settings
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser.add_argument('--use_3d', action='store_true', default=True)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=40000)
    parser.add_argument('--accumulate_grad_batches', type=int, default=8)
    parser.add_argument('--enable_flash', action='store_true', default=False)
    parser = Blip2Stage3.add_model_specific_args(parser)
    parser = Stage3DM.add_model_specific_args(parser)
    parser = SimpleUniMolModel.add_args(parser)
    args = parser.parse_args()

    args.text_max_len = 320 ############

    if args.enable_flash:
        replace_llama_attn_with_flash_attn()
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")

    print(args)
    return args


if __name__ == '__main__':
    main(get_args())

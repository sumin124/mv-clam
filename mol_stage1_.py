import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,4,5'
import argparse
import torch
from torch.utils import data
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from model.mol_blip2_stage1_ablation import Blip2Stage1
from model.unimol import SimpleUniMolModel
from model.dist_funs import MyDeepSpeedStrategy
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer, AutoModel, BertConfig
from torch_geometric.data import (Data, Dataset, DataLoader, InMemoryDataset, download_url,
                                  extract_zip)

from data_provider.d2_d3_dataset import *

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)



class Stage1DM(LightningDataModule):
    def __init__(
        self,
        num_workers = 0,
        batch_size = 256,
        root = './data_preprocess/PubChem_MAT',
        text_max_len = 128,
        pad_to_multiple = 8,
        dictionary=None,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        self.dictionary = dictionary
        self.tokenizer = tokenizer
        self.args = args

        
        if args.mode == 'pretrain':
            self.train_dataset = pt_MolDataset(root+'/pretrain/', tokenizer, text_max_len, dictionary, args.unimol_max_atoms, enriched_description=args.enriched_description).shuffle()
            print('Pretrain dataset is loaded')
            self.val_dataset = pt_MolDataset(root + '/val/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary).shuffle()
            print('Val dataset is loaded')
            self.val_dataset_match = pt_MolDataset(root + '/val/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary).shuffle()
            print('Val dataset match is loaded')
            self.test_dataset_match = pt_MolDataset(root + '/test/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary).shuffle()
            print('Test dataset match is loaded')
        elif args.mode in ['ft', 'train']:
            self.train_dataset = pt_MolDataset(root+'/train/', tokenizer, text_max_len, dictionary, args.unimol_max_atoms, enriched_description=args.enriched_description).shuffle()
            print('Train dataset is loaded')
            self.val_dataset = pt_MolDataset(root + '/val/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary).shuffle()
            print('Val dataset is loaded')
            self.val_dataset_match = pt_MolDataset(root + '/val/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary).shuffle()
            print('Val dataset match is loaded')
            self.test_dataset_match = pt_MolDataset(root + '/test/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary).shuffle()
            print('Test dataset match is loaded')
        elif args.mode == 'eval':
            self.train_dataset = pt_MolDataset(root+'/val/', tokenizer, text_max_len, dictionary, args.unimol_max_atoms, enriched_description=args.enriched_description).shuffle()
            print('Train(val) dataset is loaded')
            self.val_dataset = pt_MolDataset(root + '/val/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary).shuffle()
            print('Val dataset is loaded')
            self.val_dataset_match = pt_MolDataset(root + '/val/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary).shuffle()
            print('Val dataset match is loaded')
            self.test_dataset_match = pt_MolDataset(root + '/test/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary).shuffle()
            print('Test dataset match is loaded')
        else:
            print('Enter proper value for args.mode')
            sys.exit()

        self.val_match_loader = data.DataLoader(self.val_dataset_match, 
                                           batch_size=self.match_batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers, 
                                           pin_memory=False, 
                                           drop_last=False, 
                                           persistent_workers=True,
                                           collate_fn=MyCollater(tokenizer, self.args.text_max_len, self.dictionary.pad(), pad_to_multiple))
        self.test_match_loader = data.DataLoader(self.test_dataset_match, 
                                            batch_size=self.match_batch_size,
                                            shuffle=False,
                                            num_workers=self.num_workers, 
                                            pin_memory=False, 
                                            drop_last=False, 
                                            persistent_workers=True,
                                            collate_fn=MyCollater(tokenizer, self.args.text_max_len, self.dictionary.pad(), pad_to_multiple))
    

    def train_dataloader(self):
        loader = data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.tokenizer, self.args.text_max_len, self.dictionary.pad(), pad_to_multiple=8))
        return loader

    def val_dataloader(self):
        loader = data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.tokenizer, self.args.text_max_len, self.dictionary.pad(), pad_to_multiple=8))

        return loader



def main(args):
    print(args)
    pl.seed_everything(args.seed)

    # model
    if args.init_checkpoint:
        model = Blip2Stage1.load_from_checkpoint(args.init_checkpoint, device=args.devices, strict=False)
        print(f"loading model from {args.init_checkpoint}")
    else:
        model = Blip2Stage1(args)
    
    print('total params:', sum(p.numel() for p in model.parameters()))


    # tokenizer / dictionary
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir='../ckpts/SciBert')
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})

    dictionary = Dictionary.load('./data_provider/unimol_dict.txt')
    dictionary.add_symbol("[MASK]", is_special=True)


    # data

    dm = Stage1DM(args.num_workers, args.batch_size, args.root, args.text_max_len, 8, dictionary, tokenizer, args)
    model.val_match_loader = dm.val_match_loader
    model.test_match_loader = dm.test_match_loader

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    
    find_unused_parameters = (not args.gtm) or (not args.lm)
    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = strategies.DDPStrategy(start_method='spawn', find_unused_parameters=find_unused_parameters)
    else:
        strategy = 'auto'
#        args.devices = eval(args.devices)
        print(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
        log_every_n_steps = 100,
        limit_val_batches=1.0,
    )
    if args.mode in ['pretrain', 'ft', 'train']:
        trainer.fit(model, datamodule=dm)
        trainer.validate(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.max_epochs - 1
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = Blip2Stage1.add_model_specific_args(parser)  
    parser = SimpleUniMolModel.add_args(parser)

    parser.add_argument('--filename', type=str, default="stage1")

    parser.add_argument('--seed', type=int, default=42, help='random seed')


    parser.add_argument('--gtm', action='store_false', help='use graph-text matching or not', default=True)
    parser.add_argument('--lm', action='store_false', help='use language modeling or not', default=True)
    parser.add_argument('--lm_weight', type=int, default=2)

    parser.add_argument('--mode', type=str, default='pretrain')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser.add_argument('--enriched_description', action='store_true', default=False)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='1')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    parser.add_argument('--save_every_n_epochs', type=int, default=5)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1) 
    parser.add_argument('--batch_size', type=int, default=32)  

    parser.add_argument('--only_2d', action='store_true', default=False)
    parser.add_argument('--only_3d', action='store_true', default=False)

    # parser.add_argument('--retrieval_eval_epoch', type=int, default=5)

    args = parser.parse_args()

    args.graph_pooling = 'sum'
    args.num_workers = 4
    args.root = './data_preprocess/PubChem_MAT'
    args.text_max_len = 256
    # args.text_max_len = 600
    args.match_batch_size = 32
    # args.retrieval_eval_epoch = 1
#    args.num_query_token = 12

    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")



    main(args)


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,6,7' 
import torch
from torch.utils import data
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies, LightningDataModule
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from data_provider.d2_d3_dataset import *
from model.unimol import SimpleUniMolModel
from model.mol_blip2_stage2_ import Blip2Stage2
from model.dist_funs import MyDeepSpeedStrategy
from model.llama_flash_attention import replace_llama_attn_with_flash_attn
import pickle

import sys

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
torch.set_float32_matmul_precision('medium')


class Stage2DM(LightningDataModule):
    def __init__(
            self,
            mode: str = 'pretrain',
            num_workers: int = 0,
            batch_size: int = 256,
            root: str = './data_preprocess/PubChem_MAT',
            text_max_len: int = 128,
            pad_to_multipe: int = 8,
            dictionary=None,
            tokenizer=None,
            args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.pad_to_multiple = pad_to_multipe
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = "Below is an instruction that describes a task, paired with an input molecule. Write a response that appropriately completes the request.\n" \
                      "Instruction: Describe the input molecule.\n" \
                      "Input molecule: {} <mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol>.\n" \
                      "Response: "
        
        self.dictionary = dictionary
       # print(args)
        if self.mode in ['pretrain', 'test']:
            print('LOADING PUBCHEM DATASET...')
            
            self.train_dataset = pt_MolDataset(root+'/val/', tokenizer, text_max_len, dictionary, args.unimol_max_atoms, self.prompt, return_prompt=True, enriched_description=args.enriched_description).shuffle()
            print('Pretrain dataset is loaded')
            
            self.val_dataset = pt_MolDataset(root + '/val/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary, max_atoms=args.unimol_max_atoms, prompt = self.prompt, return_prompt = True, enriched_description=args.enriched_description).shuffle() 
            print('Val dataset is loaded')
            
            self.test_dataset = pt_MolDataset(root + '/test/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary, max_atoms=args.unimol_max_atoms, prompt=self.prompt, return_prompt=True, enriched_description=args.enriched_description).shuffle()
            print('Test dataset match is loaded')

        if self.mode == 'ft':
            print('LOADING PUBCHEM DATASET...')
            self.train_dataset = pt_MolDataset(root+'/train/', tokenizer, text_max_len, dictionary, args.unimol_max_atoms, self.prompt, return_prompt=True, enriched_description=args.enriched_description).shuffle()
            print('Pretrain dataset is loaded')
            
            self.val_dataset = pt_MolDataset(root + '/val/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary, max_atoms=args.unimol_max_atoms, prompt = self.prompt, return_prompt = True, enriched_description=args.enriched_description).shuffle() 
            print('Val dataset is loaded')
            
            self.test_dataset = pt_MolDataset(root + '/test/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary, max_atoms=args.unimol_max_atoms, prompt=self.prompt, return_prompt=True, enriched_description=args.enriched_description).shuffle()
            print('Test dataset match is loaded')


        elif self.mode == 'eval':
            print('LOADING PUBCHEM DATASET...')
            
            self.train_dataset = pt_MolDataset(root+'/pretrain/', tokenizer, text_max_len, dictionary, args.unimol_max_atoms, self.prompt, return_prompt=True, enriched_description=args.enriched_description).shuffle()
            print('Pretrain dataset is loaded')
            
            self.val_dataset = pt_MolDataset(root + '/val/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary, max_atoms=args.unimol_max_atoms, prompt = self.prompt, return_prompt = True, enriched_description=args.enriched_description).shuffle() 
            print('Val dataset is loaded')
            
            self.test_dataset = pt_MolDataset(root + '/test/', tokenizer = tokenizer, text_max_len = text_max_len, unimol_dict=dictionary, max_atoms=args.unimol_max_atoms, prompt=self.prompt, return_prompt=True, enriched_description=args.enriched_description).shuffle()
            print('Test dataset match is loaded')
            
        self.init_tokenizer(tokenizer)

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        if hasattr(self, 'pretrain_dataset'):
            self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id

    def train_dataloader(self):
        loader = data.DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            pin_memory=False,
                            drop_last=True,
                            persistent_workers=True,
                            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.dictionary.pad(), self.pad_to_multiple, self.args.num_query_token) ###### args.num_query_token argument added - 0623 kjh
                            )
        return loader

    def val_dataloader(self):
        val_loader = data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.dictionary.pad(), self.pad_to_multiple, self.args.num_query_token), ###### args.num_query_token argument added - 0623 kjh
        )
        return val_loader

    def test_dataloader(self):
        loader = data.DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.dictionary.pad(), self.pad_to_multiple, self.args.num_query_token), ###### args.num_query_token argument added - 0623 kjh
        )
        return loader



def main(args):
    pl.seed_everything(args.seed)


    if len(args.devices.split(',')) > 1:

        if args.strategy_name == 'deepspeed':
            print('deepspeed')
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            print('deepspeed_else')
            strategy = strategies.DDPStrategy(start_method='spawn')
    else:
        strategy = 'auto'
        args.devices = eval(args.devices)
        
    # model
    if args.init_checkpoint:
        model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        print('Loading from stage2_path')
        model = Blip2Stage2(args)
        ckpt = torch.load(args.stage2_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    elif args.stage1_path:
        print('Loading from stage1_path')
        model = Blip2Stage2(args)
        print(f"loading stage1 model from {args.stage1_path}")
        model.load_from_stage1_checkpoint(args.stage1_path)
    else:
        model = Blip2Stage2(args)

    print(' total params:', sum(p.numel() for p in model.parameters()))

    tokenizer = model.blip2opt.llm_tokenizer 

    dm = Stage2DM(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len, 8, model.blip2opt.dictionary, tokenizer, args)

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_last=True, 
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    
    
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
    )

        
    if args.mode == 'pretrain':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'ft':
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.test(model, datamodule=dm)
    elif args.mode == 'test':
        trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # MM settings
    parser.add_argument('--mode', type=str, default='ft')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser.add_argument('--enriched_description', action='store_true', default=False)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='1')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=2)
    parser.add_argument('--enable_flash', action='store_true', default=True)
    parser.add_argument('--agg_method', type=str, default='concat')
    parser.add_argument('--batch_size', type=int, default=5) 
    parser.add_argument('--text_max_len', type=int, default=512) 
    parser.add_argument('--inference_batch_size', type=int, default=8)
    
    parser = Blip2Stage2.add_model_specific_args(parser)
    parser = SimpleUniMolModel.add_args(parser)
    args = parser.parse_args()
    args.num_workers = 4
    args.root = './data_preprocess/PubChem_MAT' 
    if args.enable_flash:
        replace_llama_attn_with_flash_attn()
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")

    return args


if __name__ == '__main__':
    main(get_args())

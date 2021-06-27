import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import BertPolicyValue, BertMLM
from data import BertPolicyValueData, BertMLMData


def main(args):
    trainer = pl.Trainer.from_argparse_args(parser)
    callback = ModelCheckpoint(filename='{step:07d}-{val_loss:.2f}',
        monitor='val_loss', mode='min', save_top_k=1, save_last=True)
    trainer.callbacks.append(callback)

    if args.mlm:
        model = BertMLM(model_dir=args.model_dir)
        data = BertMLMData(
            dataset_dir=args.dataset_dir, batch_size=args.batch_size)
    else:
        model = BertPolicyValue(model_dir=args.model_dir)
        data = BertPolicyValueData(
            dataset_dir=args.dataset_dir, batch_size=args.batch_size)
    trainer.fit(model, datamodule=data)
    if args.model_dir and args.mlm:
        model.bert.save_pretrained(args.model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('-d', '--dataset_dir', help='Path to dataset folder',
                        required=True)
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int,
                        default=128)
    parser.add_argument('-m', '--mlm', help='Train BERT for MLM',
                        action='store_true')
    parser.add_argument('-md', '--model_dir',
                        help='Path to save/load BERT model')
    args = parser.parse_args()
    main(args)

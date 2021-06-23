import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import BertPolicyValue
from data import BertPolicyValueData


def main(args):
    trainer = pl.Trainer.from_argparse_args(parser)
    callback = ModelCheckpoint(filename='{step:07d}-{val_loss:.2f}',
        monitor='val_loss', mode='min', save_top_k=1, save_last=True)
    trainer.callbacks.append(callback)

    model = BertPolicyValue()
    data = BertPolicyValueData(
        dataset_dir=args.dataset_dir, batch_size=args.batch_size)
    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('-d', '--dataset_dir', help='Path to dataset folder',
                        required=True)
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int,
                        default=128)
    args = parser.parse_args()
    main(args)

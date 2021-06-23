import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from transformers import BertConfig, BertModel, AdamW


config = BertConfig.from_dict({
    'vocab_size': 17,  # according to board state definition
    'hidden_size': 768,
    'num_hidden_layers': 12,
    'num_attention_heads': 12,
    'intermediate_size': 3072,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
    'max_position_embeddings': 512,
    'type_vocab_size': 1,  # only one type of token_type_embedding
    'initializer_range': 0.02
})


class BertPolicyValue(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = BertModel(config)
        self.policy_head = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.Tanh(),
            nn.Linear(768 * 2, 8 * 8 * 28)
        )
        self.value_head = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.Tanh(),
            nn.Linear(768 * 2, 1),
            nn.Sigmoid()
        )

        self.loss_policy_fn = nn.CrossEntropyLoss()
        self.loss_value_fn = nn.MSELoss()

    def forward(self, input_ids, labels=None):
        features = self.bert(input_ids=input_ids)['last_hidden_state']
        policy = self.policy_head(features).mean(axis=1)
        value = self.value_head(features).mean(axis=1).squeeze(1)
        if labels is None:
            return {'policy': policy, 'value': value}
        else:
            loss_policy = self.loss_policy_fn(policy, labels['move'])
            loss_value = self.loss_value_fn(value, labels['value'])
            loss = loss_policy + loss_value
            return {'loss_policy': loss_policy, 'loss_value': loss_value,
                    'loss': loss}

    def training_step(self, batch, batch_idx):
        input_ids = batch.pop('input_ids')
        output = self(input_ids, batch)
        return {'loss': output['loss']}

    def validation_step(self, batch, batch_idx):
        input_ids = batch.pop('input_ids')
        output = self(input_ids, batch)
        for k, v in output.items():
            output[k] = v.detach().cpu().numpy()
        return output

    def validation_epoch_end(self, outputs):
        val_loss = np.mean([out['loss'] for out in outputs])
        val_loss_policy = np.mean([out['loss_policy'] for out in outputs])
        val_loss_value = np.mean([out['loss_value'] for out in outputs])
        self.log('val_loss', val_loss)
        self.log('val_loss_policy', val_loss_policy)
        self.log('val_loss_value', val_loss_value)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-5)

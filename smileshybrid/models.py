import pandas as pd
import wandb

import torch
from torch.nn import nn
from torch.utils.data import DataLoader
from transformers import AutoModel

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

class CNN_Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1000, embedding_dim)
        
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = nn.ReLU()(self.fc(x))
        return x
      


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, (input_size+output_size)//2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear((input_size+output_size)//2, output_size)
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x
      


class Transformer(nn.Module):
    """
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py
    """
    def __init__(self):
        super(Transformer, self).__init__()
        self.model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", 
                                               is_decoder=True, 
                                               add_cross_attention=True)
        self.dropout = nn.Dropout(0.5)
        self.regressor = nn.Linear(self.model.config.hidden_size, 1, bias=False)
        
    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        transformer_outputs  = self.model(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          encoder_hidden_states=encoder_hidden_states)
        
        hidden_states = transformer_outputs[0]
        logits = self.regressor(hidden_states)
        #pooled_logits = logits[range(batch_size), sequence_length]
        pooled_logits = logits.squeeze(-1)[:,-1]
        
        return pooled_logits
      
      
class SMILES_hybrid(LightningModule):
    def __init__(self):
        super(SMILES_hybrid, self).__init__()
        self.transformer = Transformer()
        self.cnn = CNN_Encoder(self.transformer.model.config.hidden_size)
        self.mlp = MLP(2048, self.transformer.model.config.hidden_size)
        #self.gnn = GATModel(mode='regression', n_tasks=1,
        #                    batch_size=10, learning_rate=0.001)
        self.L1Loss = nn.L1Loss()
        self.optimizer = AdamW(params=self.optimized_params(),
                               lr=5e-5,
                               weight_decay=1e-2)
        self.scheduler = get_cosine_schedule_with_warmup(optimizer= self.optimizer,
                                                         num_warmup_steps= 200,
                                                         num_training_steps= 1000,
                                                         num_cycles= 0.5,
                                                         )

    def forward(self, **inputs):
        batch_size, sequence_length = inputs['input_ids'].shape[:2]
        cnn_out = self.cnn(inputs['img'])
        mlp_out = self.mlp(inputs['features'])
        #gnn_out = self.gnn(inputs['features'])
        #hidden_states = torch.cat((cnn_out.unsqueeze(1), mlp_out.unsqueeze(1), gnn_out.unsqueeze(1)), dim=1)
        hidden_states = torch.cat((cnn_out.unsqueeze(1), mlp_out.unsqueeze(1)), dim=1)
        logits  = self.transformer(input_ids=inputs['input_ids'].long(),
                                   attention_mask=inputs['attention_mask'].long(),
                                   encoder_hidden_states=hidden_states)
        return logits

    def training_step(self, batch, batch_idx) -> dict:
        batch = to(batch, device="gpu")
        logits = self(**batch)
        loss = self.L1Loss(logits, batch['labels'])

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> dict:
        batch = to(batch, device="gpu")
        logits = self(**batch)
        loss = self.L1Loss(logits, batch['labels'])

        self.log("ST1_GAP(eV) MAE", loss)
        return loss

    def test_step(self, batch, batch_idx) -> dict:
        batch = to(batch, device="gpu")
        logits = self(**batch)
        return logits

    def test_epoch_end(self, outputs) -> None:
        preds = torch.cat([x for x in outputs]).detach().cpu().numpy()
        sample = pd.read_csv('./data/sample_submission.csv')
        sample['ST1_GAP(eV)'] = preds
        sample.to_csv('./data/SMILES_hybrid_submission.csv',index=False)
        
        return

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Start model training
        Args:
            train_loader (DataLoader): training data loader
            val_loader (DataLoader): validation data loader
        """
        args = {
            'gpus': 1,
            'precision': 16,
            'accumulate_grad_batches': 1,
            'val_check_interval': 1.0,
            'gradient_clip_val': 1.0,
            'max_steps':3000*5,
            "logger": WandbLogger(
                name='SMILES_hybrid',
                project="Samsung AI Challenge for Scientific Discovery",
            ),
            "callbacks": [
                LearningRateMonitor(
                    logging_interval="step",
                    log_momentum=False,
                ),
                ModelCheckpoint(
                    monitor="ST1_GAP(eV) MAE",
                    dirpath="./models",
                    filename="model.epoch={epoch:02d}.loss={valid_loss:.3f}",
                    save_top_k=5,
                    mode="min",
                ),
            ],
        }
        trainer = Trainer(
            **args,
        )

        trainer.fit(
            model=self,
            train_dataloader=train_loader,
            val_dataloaders=val_loader,
        )

    def test(
        self,
        test_loader: DataLoader,
    ) -> None:
        """
        Start model test and save
        Args:
            test_loader (DataLoader): test data loader
        """
        args = {
            'gpus': 1,
        }
        trainer = Trainer(
            **args,
        )

        trainer.test(
            model=self,
            test_dataloaders=test_loader,
        )
    


    def optimized_params(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-2,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return optimizer_grouped_parameters

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step",}]

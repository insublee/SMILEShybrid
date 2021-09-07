import pandas as pd
import numpy as np
import wandb

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
from omegaconf import DictConfig

from torchvision import models
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from smileshybrid.utils import to


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
      

class MolecularGNN(torch.nn.Module):
    """https://github.com/masashitsubaki/molecularGNN_smiles"""
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output):
        super(MolecularGNN, self).__init__()
        self.embed_fingerprint = torch.nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = torch.nn.ModuleList([torch.nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output = torch.nn.ModuleList([torch.nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = torch.nn.Linear(dim, 1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer_hidden = layer_hidden
        self.layer_output = layer_output
        # self.config = config

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        # molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors

    def mlp(self, vectors):
        """Classifier or regressor based on multilayer perceptron."""
        for i in range(self.layer_output):
            vectors = torch.relu(self.W_output[i](vectors))
        outputs = self.W_property(vectors)
        return outputs


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
    def __init__(self, config:DictConfig) -> None:
        super(SMILES_hybrid, self).__init__()
        self.config = config
        self.transformer = Transformer()
        self.cnn = CNN_Encoder(self.transformer.model.config.hidden_size)
        self.mlp = MLP(2048, self.transformer.model.config.hidden_size)
        #self.gnn = GATModel(mode='regression', n_tasks=1,
        #                    batch_size=10, learning_rate=0.001)
        self.L1Loss = nn.L1Loss()
        self.optimizer = AdamW(params=self.optimized_params(),
                               lr=self.config.train.lr,
                               weight_decay=self.config.train.weight_decay)
        self.scheduler = get_cosine_schedule_with_warmup(optimizer= self.optimizer,
                                                         num_warmup_steps= self.config.train.args.max_steps//10,
                                                         num_training_steps= self.config.train.args.max_steps,
                                                         num_cycles= 0.5,
                                                         )
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", use_fast=True)

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

        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx) -> dict:
        batch = to(batch, device="gpu")
        logits = self(**batch)
        return logits

    def test_epoch_end(self, outputs) -> None:
        preds = torch.cat([x for x in outputs]).detach().cpu().numpy()
        sample = pd.read_csv(self.config.data.sample_submission_path)
        sample['ST1_GAP(eV)'] = preds
        sample.to_csv(self.config.data.submission_path,index=False)
        
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
        
        trainer_options = {
            "logger": WandbLogger(
                name=self.config.model.id,
                project=self.config.project,
            ),
            "callbacks": [
                LearningRateMonitor(
                    logging_interval="step",
                    log_momentum=False,
                ),
                ModelCheckpoint(
                    monitor="valid_loss",
                    dirpath=self.config.train.save_path,
                    filename="model.epoch={epoch:02d}.loss={valid_loss:.3f}",
                    save_top_k=5,
                    mode="min",
                ),
            ],
        }
        trainer = Trainer(
            **trainer_options,
            **self.config.train.args,
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

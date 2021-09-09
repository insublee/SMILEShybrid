from smileshybrid.datamodule import Datamodule
from smileshybrid.models import SMILES_hybrid
from smileshybrid.utils import load_configs
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='config name in config folder')
    parser.add_argument('-c', default='baseline.yaml', help='config name in config folder (default: baseline.yaml)')
    args = parser.parse_args()
    
    # 1. create config objects
    config = load_configs("../configs/", args.c)

    # 2. create model
    model = SMILES_hybrid(config=config)

    # 3. create datasets
    dm = Datamodule(
        tokenizer=model.tokenizer,
        config=config,
    )
    # 4. start to train
    model.fit(
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
    )
    
    # 5. predict
    model.test(
        test_loader=dm.test_dataloader()
    )

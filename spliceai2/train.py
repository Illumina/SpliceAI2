import argparse
import ast
import pandas as pd
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from spliceai2.dataset import SequenceDataset
from spliceai2.model import SpliceAI2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder")
    parser.add_argument("--coord_tsv_file")
    parser.add_argument("--jxn_tsv_file")
    parser.add_argument("--tx_tsv_file")
    parser.add_argument("--fasta_tsv_file")
    parser.add_argument("--in_len", type=int)
    parser.add_argument("--out_1d_len", type=int)
    parser.add_argument("--out_2d_len", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_features", type=int)
    parser.add_argument("--num_blocks", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--num_devices", type=int)
    args = parser.parse_args()

    print(args.model_folder)
    coord_df = pd.read_csv(args.coord_tsv_file, sep="\t")
    train_coord_df = coord_df[
        (coord_df["assembly"] != "GRCh38")
        | (~coord_df["chrom"].isin([f"chr{i}" for i in range(21, 23)]))
    ]
    val_coord_df = coord_df[
        (coord_df["assembly"] == "GRCh38")
        & (coord_df["chrom"].isin([f"chr{i}" for i in range(21, 23)]))
    ]
    jxn_df = pd.read_csv(args.jxn_tsv_file, sep="\t")
    tx_df = pd.read_csv(args.tx_tsv_file, sep="\t")
    tx_df["exons"] = tx_df["exons"].map(ast.literal_eval)
    fasta_df = pd.read_csv(args.fasta_tsv_file, sep="\t")
    train_dl = DataLoader(
        SequenceDataset(
            train_coord_df,
            jxn_df,
            tx_df,
            fasta_df,
            args.in_len,
            args.out_1d_len,
            args.out_2d_len,
            augment=True
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    val_dl = DataLoader(
        SequenceDataset(
            val_coord_df,
            jxn_df,
            tx_df,
            fasta_df,
            args.in_len,
            args.out_1d_len,
            args.out_2d_len
        ),
        batch_size=args.batch_size,
        num_workers=2
    )

    model = SpliceAI2(
        args.num_features,
        args.num_blocks,
        crop_len=args.in_len - args.out_1d_len
    )
    checkpoint = ModelCheckpoint(
        dirpath=args.model_folder, filename="model", monitor="val_loss"
    )
    trainer = Trainer(
        default_root_dir=args.model_folder,
        max_epochs=args.num_epochs,
        log_every_n_steps=len(train_dl) // args.num_devices,
        val_check_interval=len(train_dl) // args.num_devices,
        sync_batchnorm=args.num_devices > 1,
        callbacks=[checkpoint],
        precision="bf16-mixed",
        gradient_clip_val=1e-4
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()

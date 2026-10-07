import argparse
import pandas as pd
from torch.utils.data import DataLoader
from lightning import Trainer
import torch
from spliceai2.dataset import VariantDataset
from spliceai2.model import SpliceAI2, VariantAnnotator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folders", nargs="+")
    parser.add_argument("--model_suffix", default="spliceai2")
    parser.add_argument("--var_tsv_file")
    parser.add_argument("--fasta_file")
    parser.add_argument("--in_len", type=int, default=196608)
    parser.add_argument("--out_2d_len", type=int, default=1024)
    parser.add_argument("--assembly", default="GRCh38")
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()

    print(args.model_folders, args.model_suffix)
    var_df = pd.read_csv(args.var_tsv_file, sep="\t")
    var_df["assembly"] = args.assembly
    var_dl = DataLoader(
        VariantDataset(var_df, args.fasta_file, args.in_len), num_workers=2
    )

    models = [
        SpliceAI2.load_from_checkpoint(f"{model_folder}/model.ckpt")
        for model_folder in args.model_folders
    ]
    var_model = VariantAnnotator(models, args.out_2d_len)
    trainer = Trainer(logger=False, precision="16-mixed")
    if not args.no_compile:
        var_model = torch.compile(var_model)
    preds = trainer.predict(var_model, dataloaders=var_dl)
    result = torch.cat([p[0] for p in preds]).cpu().numpy()
    col = preds[0][1]

    var_df = var_df.join(pd.DataFrame(result, columns=col))
    var_df = var_df.drop(columns="assembly")
    var_df["spliceai2_summary_score"] = var_df[[
        "donor_gain_delta_score_0",
        "donor_loss_delta_score_0",
        "acceptor_gain_delta_score_0",
        "acceptor_loss_delta_score_0",
    ]].max(axis=1)
    var_df = var_df.round({c: 2 for c in var_df.columns if "score" in c})
    var_df = var_df.astype({c: "int32" for c in var_df.columns if "dist" in c})
    var_df.to_csv(
        f"{args.var_tsv_file}.{args.model_suffix}", sep="\t", index=False
    )


if __name__ == "__main__":
    main()

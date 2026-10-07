import numpy as np
from torch.utils.data import Dataset
import pyfaidx
import pandas as pd


def _one_hot_encode(nuc_seq, assembly):
    channel_map = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "GRCh38": 4,
        "NHGRI_mPanTro3-v2.0": 5,
        "Panubis1.0": 6,
        "T2T-MFA8v1.1": 7,
        "Mmul_10": 8,
        "ARS-UCD2.0": 9,
        "ARS-UI_Ramb_v3.0": 10,
        "Sscrofa11.1": 11,
        "GRCm38": 12,
        "GRCr8": 13
    }
    x = np.zeros((14, len(nuc_seq)))
    x[[channel_map.get(n, 0) for n in nuc_seq], np.arange(len(nuc_seq))] = 1
    x[channel_map.get(assembly, 4)] = 1
    return x


class SequenceDataset(Dataset):
    def __init__(
            self,
            coord_df,
            jxn_df,
            tx_df,
            fasta_df,
            in_len,
            out_1d_len,
            out_2d_len,
            augment=False
    ):
        self._coord_df = coord_df
        self._jxn_grp = jxn_df.groupby("gene_id")
        self._tx_grp = tx_df.groupby("gene_id")
        self._fasta_df = fasta_df
        self._fasta = None
        self._in_len = in_len
        self._out_1d_len = out_1d_len
        self._out_2d_len = out_2d_len
        self._augment = augment

    def __len__(self):
        return len(self._coord_df)

    def __getitem__(self, smpl_idx):
        x = np.zeros((14, self._in_len))
        y_ss = np.zeros((2, self._out_1d_len))
        y_jxn = np.zeros((self._out_2d_len, self._out_2d_len))
        y_tx = np.zeros((self._out_2d_len, self._out_2d_len))
        mask_ss = np.zeros(self._out_1d_len)
        mask_jxn = np.zeros((self._out_2d_len, self._out_2d_len))
        mask_tx = np.zeros((self._out_2d_len, self._out_2d_len))
        y_2d_idxs = np.zeros(self._out_2d_len, dtype=np.int64)

        assembly = self._coord_df.iloc[smpl_idx]["assembly"]
        chrom = self._coord_df.iloc[smpl_idx]["chrom"]
        strand = self._coord_df.iloc[smpl_idx]["strand"]
        strand = -1 if strand in ("-", -1) else 1
        gene_id = self._coord_df.iloc[smpl_idx]["gene_id"]
        gene_start = self._coord_df.iloc[smpl_idx]["gene_start"]
        gene_end = self._coord_df.iloc[smpl_idx]["gene_end"]
        pos = self._coord_df.iloc[smpl_idx]["pos"]
        if self._augment:
            pos_aug_len = (gene_start - gene_end) % self._out_1d_len
            pos += np.random.choice(pos_aug_len + 1) - pos_aug_len // 2

        if self._fasta is None:
            self._fasta = {
                assembly: pyfaidx.Fasta(path) for assembly, path in
                zip(self._fasta_df["assembly"], self._fasta_df["path"])
            }
        try:
            nuc_seq = self._fasta[assembly][chrom][
                (pos - 1) - self._in_len // 2:(pos - 1) + self._in_len // 2
            ].seq.upper()
        except pyfaidx.FetchError:
            nuc_seq = ""

        if len(nuc_seq) == self._in_len:
            gene_jxn_df = self._jxn_grp.get_group(gene_id).copy()
            gene_ss_df = pd.DataFrame({
                col: gene_jxn_df.groupby(col)["jxn_count"].sum()
                for col in ["donor", "acceptor"]
            })
            gene_jxn_tbl = gene_jxn_df.pivot_table(
                index="donor", columns="acceptor", values="jxn_count"
            )
            x = _one_hot_encode(nuc_seq, assembly)
            y_1d_pos = np.arange(
                pos - self._out_1d_len // 2, pos + self._out_1d_len // 2
            )
            y_ss = (
                gene_ss_df.
                div(gene_ss_df.max().max()).
                reindex(y_1d_pos).
                fillna(0).
                pow(0.33).
                transpose().
                values
            )
            y_eps = 1e-8 * np.sin(np.arange(self._out_1d_len))
            y_2d_idxs = np.sort(
                (y_ss.sum(axis=0) + y_eps).argsort()[-self._out_2d_len:]
            )
            y_2d_pos = y_1d_pos[y_2d_idxs]
            y_jxn = (
                gene_jxn_tbl.
                div(gene_ss_df.max().max()).
                reindex(index=y_2d_pos, columns=y_2d_pos).
                fillna(0).
                pow(0.33).
                values
            )
            mask_ss = np.isin(y_1d_pos, np.arange(gene_start, gene_end))
            mask_jxn = np.outer(mask_ss[y_2d_idxs], mask_ss[y_2d_idxs])

        if len(nuc_seq) == self._in_len and gene_id in self._tx_grp.groups:
            gene_tx_df = self._tx_grp.get_group(gene_id).copy()
            gene_tx_df = gene_tx_df.sort_values("tx_count", ascending=False)
            tx_start = gene_tx_df.iloc[0]["exons"][0][0]
            tx_end = gene_tx_df.iloc[0]["exons"][-1][1]
            for i in range(len(gene_tx_df)):
                exons = np.asarray(gene_tx_df.iloc[i]["exons"])
                tx_jxn_tbl = pd.DataFrame(
                    np.eye(len(exons) - 1) * gene_tx_df.iloc[i]["tx_count"],
                    index=exons[:-1, 1] + 1,
                    columns=exons[1:, 0] - 1
                )
                y_tx += (
                    tx_jxn_tbl.
                    div(gene_tx_df["tx_count"].sum()).
                    reindex(index=y_2d_pos, columns=y_2d_pos).
                    fillna(0).
                    values
                )
            _mask_tx = np.isin(y_1d_pos, np.arange(tx_start, tx_end))
            mask_tx = np.outer(_mask_tx[y_2d_idxs], _mask_tx[y_2d_idxs])

        if strand == -1:
            x[:4] = x[:4][::-1, ::-1]
            y_ss = y_ss[:, ::-1]
            y_jxn = y_jxn[::-1, ::-1]
            y_tx = y_tx[::-1, ::-1].T
            mask_ss = mask_ss[::-1]
            mask_jxn = mask_jxn[::-1, ::-1]
            mask_tx = mask_tx[::-1, ::-1].T
            y_2d_idxs = self._out_1d_len - 1 - y_2d_idxs[::-1]

        return (
            x.astype("uint8"),
            y_ss.astype("float32"),
            y_jxn.astype("float32"),
            y_tx.astype("float32"),
            mask_ss.astype("uint8"),
            mask_jxn.astype("uint8"),
            mask_tx.astype("uint8"),
            y_2d_idxs.astype("int64")
        )


class VariantDataset(Dataset):
    def __init__(self, var_df, fasta_file, in_len):
        self._var_df = var_df
        self._fasta_file = fasta_file
        self._fasta = None
        self._in_len = in_len

    def __len__(self):
        return len(self._var_df)

    def __getitem__(self, smpl_idx):
        x_ref = np.zeros((14, self._in_len))
        x_alt = np.zeros((14, self._in_len))

        assembly = self._var_df.iloc[smpl_idx]["assembly"]
        chrom = self._var_df.iloc[smpl_idx]["chrom"]
        pos = self._var_df.iloc[smpl_idx]["pos"]
        ref = self._var_df.iloc[smpl_idx]["ref"]
        alt = self._var_df.iloc[smpl_idx]["alt"]
        strand = self._var_df.iloc[smpl_idx]["strand"]
        strand = -1 if strand in ("-", -1) else 1

        if self._fasta is None:
            self._fasta = pyfaidx.Fasta(self._fasta_file)
        nuc_seq = self._fasta[chrom][
            (pos - 1) - self._in_len // 2:
            (pos - 1) + self._in_len // 2 + max(0, len(ref) - len(alt))
        ].seq.upper()

        if len(nuc_seq) == self._in_len + max(0, len(ref) - len(alt)):
            assert nuc_seq[self._in_len // 2:][:len(ref)] == ref
            nuc_seq_ref = nuc_seq[:self._in_len]
            nuc_seq_alt = (
                nuc_seq[:self._in_len // 2]
                + alt
                + nuc_seq[self._in_len // 2 + len(ref):]
            )[:self._in_len]
            x_ref = _one_hot_encode(nuc_seq_ref, assembly)
            x_alt = _one_hot_encode(nuc_seq_alt, assembly)
        if strand == -1:
            x_ref[:4] = x_ref[:4][::-1, ::-1]
            x_alt[:4] = x_alt[:4][::-1, ::-1]

        return (
            x_ref.astype("uint8"),
            x_alt.astype("uint8"),
            strand,
            len(ref),
            len(alt)
        )

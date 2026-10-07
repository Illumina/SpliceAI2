from torch import nn
import torch
from lightning import LightningModule
from torch.utils import checkpoint
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from spliceai2.muon import Muon


class MetaFormer(nn.Module):
    def __init__(self, num_features, kernel_size, dilation):
        super().__init__()
        self._token_norm = nn.BatchNorm1d(num_features)
        self._token_mixer = nn.Conv1d(
            num_features,
            num_features,
            kernel_size,
            padding="same",
            dilation=dilation,
            groups=num_features
        )
        self._ffn_norm = nn.BatchNorm1d(num_features)
        self._ffn_proj_up = nn.Conv1d(num_features, 4 * num_features, 1)
        self._ffn_act = nn.ReLU()
        self._ffn_proj_down = nn.Conv1d(4 * num_features, num_features, 1)

    def forward(self, x):
        out = x
        x = self._token_norm(out)
        x = self._token_mixer(x)
        out = out + x
        x = self._ffn_norm(out)
        x = self._ffn_proj_up(x)
        x = self._ffn_act(x)
        x = self._ffn_proj_down(x)
        out = out + x
        return out


class Attention(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self._token_norm = nn.LayerNorm(num_features)
        self._query = nn.Conv1d(num_features, num_features, 1)
        self._key = nn.Conv1d(num_features, num_features, 1)
        self._num_features = num_features

    def forward(self, x, token_idxs):
        x = x.permute(0, 2, 1)
        x = self._token_norm(x)
        x = x.permute(0, 2, 1)
        out = (self._num_features ** -0.5) * torch.einsum(
            "bcl,bcm->blm",
            self._rope(self._query(x), token_idxs),
            self._rope(self._key(x), token_idxs)
        )
        out -= out.clone().fill_(1e4).tril()
        return out

    def _rope(self, x, token_idxs):
        freqs = 1e4 ** -(
            torch.arange(0, self._num_features, 2, device=x.device)
            / self._num_features
        )
        angle_mtrx = token_idxs.unsqueeze(1) * freqs.unsqueeze(1).unsqueeze(0)
        sin, cos = angle_mtrx.sin(), angle_mtrx.cos()
        x1, x2 = x[:, 0::2], x[:, 1::2]
        out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=1)
        return out


class SpliceAI2(LightningModule):
    def __init__(
            self,
            num_features,
            num_blocks,
            kernel_size=5,
            dilation=tuple(2 ** (b // 2) for b in range(32)),
            pool=tuple(2 ** (b // 4) for b in range(32)),
            crop_len=0
    ):
        super().__init__()
        self._embed = nn.Conv1d(14, num_features, 1)
        self._downsamplers = nn.ModuleList([
            nn.AvgPool1d(pool[b]) for b in range(num_blocks)
        ])
        self._blocks = nn.ModuleList([
            MetaFormer(num_features, kernel_size, dilation[b])
            for b in range(num_blocks)
        ])
        self._upsamplers = nn.ModuleList([
            nn.Upsample(scale_factor=pool[b], mode="linear")
            for b in range(num_blocks)
        ])
        self._ss_heads = nn.ModuleList([
            nn.Conv1d(num_features, 2, 1) for _ in range(num_blocks)
        ])
        self._jxn_heads = nn.ModuleList([
            Attention(num_features) for _ in range(num_blocks)
        ])

        self._num_blocks = num_blocks
        self._crop_len = crop_len
        self._initialize_params()
        self.save_hyperparameters()

    def forward(self, x):
        ftrs = []
        x = self._embed(x)
        for b in range(self._num_blocks):
            x = self._downsamplers[b](x)
            x = checkpoint.checkpoint(self._blocks[b], x, use_reentrant=False)
            x = self._upsamplers[b](x)
            ftrs.append(
                x[..., self._crop_len // 2:-self._crop_len // 2 or None]
            )
        return ftrs

    def forward_1d(self, ftrs):
        out_ss = 0.0
        for b in range(self._num_blocks):
            out_ss += (1 / self._num_blocks) * self._ss_heads[b](ftrs[b])
        return out_ss

    def forward_2d(self, ftrs, y_2d_idxs):
        out_jxn = 0.0
        for b in range(self._num_blocks):
            out_jxn += (1 / self._num_blocks) * self._jxn_heads[b](
                ftrs[b].gather(
                    2, y_2d_idxs.unsqueeze(1).repeat(1, ftrs[b].shape[1], 1)
                ),
                y_2d_idxs
            )
        return out_jxn

    def training_step(self, batch):
        train_loss = self._step(batch)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch):
        val_loss = self._step(batch)
        self.log("val_loss", val_loss, sync_dist=True)

    def configure_optimizers(self):
        muon_params = [
            param for name_, param in self.named_parameters()
            if "ffn" in name_ and param.ndim >= 2
        ]
        adam_params = [
            param for name_, param in self.named_parameters()
            if "ffn" not in name_ or param.ndim == 1
        ]
        param_groups = [
            {"params": muon_params, "use_muon": True},
            {"params": adam_params, "use_muon": False}
        ]
        optimizer = Muon(param_groups)
        scheduler = MultiStepLR(
            optimizer, milestones=[0.8 * self.trainer.max_epochs]
        )
        return [optimizer], [scheduler]

    def _initialize_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def _step(self, batch):
        x, y_ss, y_jxn, y_tx, mask_ss, mask_jxn, mask_tx, y_2d_idxs = batch
        ftrs = self.forward(x.float())
        out_ss = self.forward_1d(ftrs).float()
        out_jxn = self.forward_2d(ftrs, y_2d_idxs).float()
        out_tx = (
            out_ss[:, 0].gather(1, y_2d_idxs).unsqueeze(2)
            + out_jxn
            + out_ss[:, 1].gather(1, y_2d_idxs).unsqueeze(1)
        )
        out_tx = out_tx.masked_fill(mask_tx.float() == 0, -1e4)

        loss_ss = F.binary_cross_entropy_with_logits(
            out_ss, y_ss, weight=mask_ss.float().unsqueeze(1)
        )
        loss_jxn = F.binary_cross_entropy_with_logits(
            out_jxn, y_jxn, weight=mask_jxn.float()
        )
        log_path = (y_tx * mask_tx.float() * out_tx).sum(dim=(1, 2))
        log_z = torch.zeros(
            y_tx.shape[0], y_tx.shape[1], device=y_tx.device, dtype=y_tx.dtype
        )
        for i in range(1, y_tx.shape[1]):
            log_z[:, i] = torch.logsumexp(
                torch.cat(
                    [log_z[:, i - 1:i], log_z[:, :i] + out_tx[:, :i, i]],
                    dim=1
                ),
                dim=1
            )
        loss_tx = (log_z[:, -1] - log_path).mean()
        loss = loss_ss + loss_jxn + 1e-5 * loss_tx
        return loss


class VariantAnnotator(LightningModule):
    def __init__(self, models, out_2d_len, k=10):
        super().__init__()
        self._models = nn.ModuleList(models)
        self._out_1d_len = None
        self._out_2d_len = out_2d_len
        self._k = k

    def forward(self, x_ref, x_alt, strand, ref_len, alt_len):
        ftrs_ref = [model.forward(x_ref) for model in self._models]
        ftrs_alt = [model.forward(x_alt) for model in self._models]
        if self._out_1d_len is None:
            self._out_1d_len = ftrs_ref[0][0].shape[2]

        idxs = torch.arange(self._out_1d_len, device=ftrs_ref[0][0].device)
        dist = torch.where(
            strand == -1,
            self._out_1d_len // 2 - 1 - idxs,
            idxs - self._out_1d_len // 2
        )
        mask_1d_ref = (dist < ref_len) | (dist >= alt_len)
        mask_1d_alt = (dist < alt_len) | (dist >= ref_len)
        dist_1d_ref = mask_1d_ref * (
            dist - (dist >= alt_len) * (alt_len - ref_len).clamp(min=0)
        ) + ~mask_1d_ref * (ref_len - 1)
        dist_1d_alt = mask_1d_alt * (
            dist - (dist >= ref_len) * (ref_len - alt_len).clamp(min=0)
        ) + ~mask_1d_alt * (alt_len - 1)
        out_1d_idxs_ref = torch.where(
            strand == -1,
            self._out_1d_len // 2 - 1 - dist_1d_ref,
            dist_1d_ref + self._out_1d_len // 2
        )
        out_1d_idxs_alt = torch.where(
            strand == -1,
            self._out_1d_len // 2 - 1 - dist_1d_alt,
            dist_1d_alt + self._out_1d_len // 2
        )
        prob_ss_ref = torch.stack([
            model.forward_1d(ftrs).sigmoid().gather(
                2, out_1d_idxs_ref.unsqueeze(1).repeat(1, 2, 1)
            ) * mask_1d_ref.unsqueeze(1)
            for model, ftrs in zip(self._models, ftrs_ref)
        ]).mean(dim=0).float()
        prob_ss_alt = torch.stack([
            model.forward_1d(ftrs).sigmoid().gather(
                2, out_1d_idxs_alt.unsqueeze(1).repeat(1, 2, 1)
            ) * mask_1d_alt.unsqueeze(1)
            for model, ftrs in zip(self._models, ftrs_alt)
        ]).mean(dim=0).float()

        subset_idxs = (
            torch.max(prob_ss_ref, prob_ss_alt).
            amax(dim=1).
            topk(self._out_2d_len, dim=1)[1].
            sort(dim=1).
            values
        )
        mask_2d_ref = mask_1d_ref.gather(1, subset_idxs)
        mask_2d_alt = mask_1d_alt.gather(1, subset_idxs)
        dist_2d_ref = dist_1d_ref.gather(1, subset_idxs)
        out_2d_idxs_ref = out_1d_idxs_ref.gather(1, subset_idxs)
        out_2d_idxs_alt = out_1d_idxs_alt.gather(1, subset_idxs)
        prob_jxn_ref = torch.stack([
            model.forward_2d(ftrs, out_2d_idxs_ref).sigmoid()
            * mask_2d_ref.unsqueeze(1) * mask_2d_ref.unsqueeze(2)
            for model, ftrs in zip(self._models, ftrs_ref)
        ]).mean(dim=0).float()
        prob_jxn_alt = torch.stack([
            model.forward_2d(ftrs, out_2d_idxs_alt).sigmoid()
            * mask_2d_alt.unsqueeze(1) * mask_2d_alt.unsqueeze(2)
            for model, ftrs in zip(self._models, ftrs_alt)
        ]).mean(dim=0).float()

        return (
            prob_ss_ref,
            prob_ss_alt,
            dist_1d_ref,
            prob_jxn_ref,
            prob_jxn_alt,
            dist_2d_ref
        )

    def predict_step(self, batch):
        x_ref, x_alt, strand, ref_len, alt_len = batch
        (
            prob_ss_ref,
            prob_ss_alt,
            dist_1d_ref,
            prob_jxn_ref,
            prob_jxn_alt,
            dist_2d_ref
        ) = self.forward(
            x_ref.float(),
            x_alt.float(),
            strand.unsqueeze(1),
            ref_len.unsqueeze(1),
            alt_len.unsqueeze(1)
        )

        result, col = zip(
            self._extract_topk(
                prob_ss_ref, prob_ss_alt, dist_1d_ref, "donor_gain"
            ),
            self._extract_topk(
                prob_ss_ref, prob_ss_alt, dist_1d_ref, "donor_loss"
            ),
            self._extract_topk(
                prob_ss_ref, prob_ss_alt, dist_1d_ref, "acceptor_gain"
            ),
            self._extract_topk(
                prob_ss_ref, prob_ss_alt, dist_1d_ref, "acceptor_loss"
            ),
            self._extract_topk(
                prob_jxn_ref, prob_jxn_alt, dist_2d_ref, "jxn_gain"
            ),
            self._extract_topk(
                prob_jxn_ref, prob_jxn_alt, dist_2d_ref, "jxn_loss"
            )
        )
        result = torch.cat(result, dim=1)
        col = sum(col, [])
        return result, col

    def _extract_topk(self, prob_ref, prob_alt, dist, effect_type):
        if "donor" in effect_type:
            prob_ref = prob_ref[:, 0]
            prob_alt = prob_alt[:, 0]
        elif "acceptor" in effect_type:
            prob_ref = prob_ref[:, 1]
            prob_alt = prob_alt[:, 1]
        elif "jxn" in effect_type:
            prob_ref = prob_ref.flatten(start_dim=1)
            prob_alt = prob_alt.flatten(start_dim=1)
        effect_sign = 1 if "gain" in effect_type else -1
        prob_delta = effect_sign * (prob_alt - prob_ref)
        topk_delta, topk_idxs = prob_delta.topk(self._k, dim=1)
        topk_ref = prob_ref.gather(1, topk_idxs)
        topk_alt = prob_alt.gather(1, topk_idxs)

        if "donor" in effect_type or "acceptor" in effect_type:
            topk_dist = dist.gather(1, topk_idxs)
            result = torch.cat(
                [topk_delta, topk_ref, topk_alt, topk_dist], dim=1
            )
            col = [
                f"{effect_type}_{r}_{k}"
                for r in ("delta_score", "ref_score", "alt_score", "dist")
                for k in range(self._k)
            ]
            return result, col
        if "jxn" in effect_type:
            topk_donor_dist = dist.gather(1, topk_idxs // self._out_2d_len)
            topk_acceptor_dist = dist.gather(1, topk_idxs % self._out_2d_len)
            result = torch.cat(
                [
                    topk_delta,
                    topk_ref,
                    topk_alt,
                    topk_donor_dist,
                    topk_acceptor_dist
                ],
                dim=1
            )
            col = [
                f"{effect_type}_{r}_{k}"
                for r in (
                    "delta_score",
                    "ref_score",
                    "alt_score",
                    "donor_dist",
                    "acceptor_dist"
                )
                for k in range(self._k)
            ]
            return result, col
        return None

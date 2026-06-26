import torch
import torch.nn as nn

from models.utils import *
from models.cfg_sampler import ClassifierFreeSampleModel
from models.blocks import *
from utils.utils import *

from models.gaussian_diffusion import (
    MotionDiffusion,
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)


class MotionEncoder(nn.Module):
    """
    Keep original class (unchanged) for compatibility.
    NOTE: original may have dim mismatch if used with x[..., :-4].
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_feats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.input_feats * 2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=2000)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)

    def forward(self, batch):
        x, mask = batch["motions"], batch["mask"]
        B, T, D = x.shape

        x = x.reshape(B, T, 2, -1)[..., :-4].reshape(B, T, -1)

        x_emb = self.embed_motion(x)

        emb = torch.cat([self.query_token[torch.zeros(B, dtype=torch.long, device=x.device)][:, None], x_emb], dim=1)

        seq_mask = (mask > 0.5)
        token_mask = torch.ones((B, 1), dtype=bool, device=x.device)
        valid_mask = torch.cat([token_mask, seq_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        motion_emb = self.out(h[:, 0])

        batch["motion_emb"] = motion_emb
        return batch


class SourceMotionEncoder(nn.Module):
    """
    Encode source motions (B,T,2*INPUT_DIM) into a compact embedding (B,512).
    Uses (INPUT_DIM-4)*2 because process_motion_np appends 4 dims per person at the end.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_feats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear((self.input_feats - 4) * 2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=2000)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)

    def forward(self, motions, mask):
        """
        motions: (B,T,2*INPUT_DIM)
        mask:   (B,T)  1=valid 0=pad
        """
        B, T, D = motions.shape
        x = motions.reshape(B, T, 2, -1)[..., :-4].reshape(B, T, -1)  # (B,T,(INPUT_DIM-4)*2)

        x_emb = self.embed_motion(x)  # (B,T,latent)
        q = self.query_token[torch.zeros(B, dtype=torch.long, device=motions.device)][:, None]
        emb = torch.cat([q, x_emb], dim=1)  # (B,T+1,latent)

        token_mask = torch.ones((B, 1), dtype=torch.bool, device=motions.device)
        valid_mask = torch.cat([token_mask, (mask > 0.5)], dim=1)  # (B,T+1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        source_emb = self.out(h[:, 0])  # (B,512)
        return source_emb


class TIMotionDenoiser(nn.Module):
    def __init__(self,
                 input_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 cfg_weight=0.,
                 **kargs):
        super().__init__()

        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim

        self.text_emb_dim = 768
        self.source_emb_dim = 512

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)
        self.source_embed = nn.Linear(self.source_emb_dim, self.latent_dim)

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, dropout, num_layers + 1)]
        for i in range(num_layers):
            self.blocks.append(
                TIMotionTransformerBlock(
                    num_heads=num_heads,
                    latent_dim=latent_dim,
                    dropout=dpr[i],
                    ff_size=ff_size,
                    num_layers=num_layers,
                    cur_layer=i,
                    LPA=kargs['cfg'].get('LPA', False),
                    cfg=kargs['cfg']
                )
            )

        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, mask=None, cond=None, source_emb=None):
        """
        x: (B,T,2*D) where D=input_feats
        cond: (B,768)
        source_emb: (B,512)
        """
        B, T = x.shape[0], x.shape[1]
        x_a, x_b = x[..., :self.input_feats], x[..., self.input_feats:]

        if mask is not None:
            mask = mask[..., 0]  # (B,T)

        emb = self.embed_timestep(timesteps) + self.text_embed(cond)
        if source_emb is not None:
            emb = emb + self.source_embed(source_emb)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(B, T, device=x_a.device)
        key_padding_mask = ~(mask > 0.5)

        for block in self.blocks:
            h_a, h_b = block(h_a_prev, h_b_prev, emb, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b

        output_a = self.out(h_a)
        output_b = self.out(h_b)
        output = torch.cat([output_a, output_b], dim=-1)
        return output


class TIMotionDiffusion(nn.Module):
    def __init__(self, cfg, sampling_strategy="ddim50"):
        super().__init__()
        self.cfg = cfg
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP

        self.cfg_weight = cfg.CFG_WEIGHT
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampler = cfg.SAMPLER
        self.sampling_strategy = sampling_strategy

        self.net = TIMotionDenoiser(
            self.nfeats,
            self.latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation,
            cfg_weight=self.cfg_weight,
            cfg=cfg
        )

        # NEW: source encoder
        self.motion_encoder = SourceMotionEncoder(cfg)

        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)

        timestep_respacing = [self.diffusion_steps]
        self.diffusion = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )
        self.sampler = create_named_schedule_sampler(self.sampler, self.diffusion)

    def mask_cond(self, cond, cond_mask_prob=0.1, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond), torch.zeros((bs, 1), device=cond.device)
        elif cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask_prob).view([bs] + [1] * (len(cond.shape) - 1))
            # return masked_cond, keep_mask(1=keep,0=drop)
            keep = (1. - mask)
            return cond * keep, keep
        else:
            return cond, None

    def generate_src_mask(self, T, length):
        length = length.detach().cpu().long().clamp(min=0, max=T)
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(int(length[i].item()), T):
                    src_mask[i, j, p] = 0
        return src_mask

    def compute_loss(self, batch):
        """
        batch["motions"]  : targets (B,T,2*D)
        batch["sources"] : sources (B,T,2*D)
        batch["motion_lens"] : target lens (B,)
        batch["source_lens"] : source lens (B,)
        batch["cond"] : text cond (B,768)
        """
        cond = batch["cond"]
        x_start = batch["motions"]       # target
        sources = batch["sources"]       # source
        B, T = x_start.shape[:2]

        # text cfg dropout (and use same keep mask for source_emb to avoid "source leak")
        cond, keep_mask = self.mask_cond(cond, 0.1)  # keep_mask: (B,1) or None

        # source mask -> source embedding
        src_T = sources.shape[1]
        src_mask = self.generate_src_mask(src_T, batch["source_lens"]).to(x_start.device)  # (B,Ts,2)
        src_mask_1 = src_mask[..., 0]  # (B,Ts)
        source_emb = self.motion_encoder(sources, src_mask_1)  # (B,512)

        if keep_mask is not None:
            # keep_mask: 1=keep,0=drop
            source_emb = source_emb * keep_mask.view(B, 1)

        # target seq mask
        tgt_mask = self.generate_src_mask(T, batch["motion_lens"]).to(x_start.device)

        t, _ = self.sampler.sample(B, x_start.device)
        output = self.diffusion.training_losses(
            model=self.net,
            x_start=x_start,
            t=t,
            mask=tgt_mask,
            t_bar=self.cfg.T_BAR,
            cond_mask=keep_mask,
            model_kwargs={
                "mask": tgt_mask,
                "cond": cond,
                "source_emb": source_emb,
            },
        )
        return output

    def forward(self, batch):
        """
        Sampling: x_start=None as requested.
        """
        cond = batch["cond"]
        sources = batch["sources"]
        B = cond.shape[0]
        T = int(batch["motion_lens"][0])
        src_T = sources.shape[1]

        # source emb
        src_mask = self.generate_src_mask(src_T, batch["source_lens"]).to(sources.device)
        source_emb = self.motion_encoder(sources, src_mask[..., 0])

        timestep_respacing = self.sampling_strategy
        self.diffusion_test = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.cfg_model = ClassifierFreeSampleModel(self.net, self.cfg_weight)

        output = self.diffusion_test.ddim_sample_loop(
            self.cfg_model,
            (B, T, self.nfeats * 2),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "mask": None,
                "cond": cond,
                "source_emb": source_emb,
            },
            x_start=None
        )
        return {"output": output}

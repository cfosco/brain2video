import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from dalle2_pytorch.dalle2_pytorch import NoiseScheduler
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tqdm import tqdm
from rotary_embedding_torch import RotaryEmbedding


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def is_float_dtype(dtype):
    return any(
        dtype == float_dtype
        for float_dtype in (
            torch.float64,
            torch.float32,
            torch.float16,
            torch.bfloat16,
        )
    )


def l2norm(t):
    return F.normalize(t, dim=-1)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def eval_decorator(fn):
    def inner(model: nn.Module, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        expansion_factor: float = 2.0,
        depth: int = 2,
        norm: bool = False,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)
        norm_fn = lambda: nn.LayerNorm(hidden_dim) if norm else nn.Identity()

        layers = [nn.Sequential(nn.Linear(dim_in, hidden_dim), nn.SiLU(), norm_fn())]

        layers.extend(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), norm_fn())
            for _ in range(depth - 1)
        )
        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x.float())


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        dtype, device = x.dtype, x.device
        assert is_float_dtype(dtype), "input to sinusoidal pos emb must be a float type"

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = rearrange(x, "i -> i 1") * rearrange(emb, "j -> 1 j")
        return torch.cat((emb.sin(), emb.cos()), dim=-1).type(dtype)


class LayerNorm(nn.Module):
    def __init__(
        self, dim: int, eps: float = 1e-5, fp16_eps: float = 1e-3, stable: bool = False
    ):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach()

        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class RelPosBias(nn.Module):
    def __init__(self, heads: int = 8, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )
        return torch.where(is_small, n, val_if_large)

    def forward(self, i, j, *, device):
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, "i j h -> h i j")


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


def FeedForward(
    dim: int,
    mult: int = 4,
    dropout: float = 0.0,
    post_activation_norm: bool = False,
):
    inner_dim = int(mult * dim)

    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_head=64,
        heads: int = 8,
        dropout: float = 0.0,
        causal: bool = False,
        rotary_emb=None,
        cosine_sim: bool = True,
        cosine_sim_scale: float = 16,
    ):
        super().__init__()
        self.scale = cosine_sim_scale if cosine_sim else (dim_head**-0.5)
        self.cosine_sim = cosine_sim

        self.heads = heads
        inner_dim = dim_head * heads
        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2)

        self.rotary_emb = rotary_emb

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor, mask=None, attn_bias=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        q = q * self.scale

        # rotary positional embedding

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        # add null key / value for cfg in prior net

        nk, nv = map(
            lambda t: repeat(t, "d -> b 1 d", b=b), self.null_kv.unbind(dim=-2)
        )
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # whether to use cosine sime

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # calculate query / key similarities

        sim = torch.einsum("b h i d, b j d -> b h i j", q, k)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(
                j - i + 1
            )
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        # aggregate

        out = torch.einsum("b h i j, b j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        norm_in=False,
        norm_out=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
        final_proj=True,
        normformer=False,
        rotary_emb=True,
        causal: bool = True,
    ):
        super().__init__()
        self.init_norm = (
            LayerNorm(dim) if norm_in else nn.Identity()
        )  # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads=heads)

        rotary_emb = RotaryEmbedding(dim=min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            causal=causal,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_emb=rotary_emb,
                        ),
                        FeedForward(
                            dim=dim,
                            mult=ff_mult,
                            dropout=ff_dropout,
                            post_activation_norm=normformer,
                        ),
                    ]
                )
            )

        self.norm = (
            LayerNorm(dim, stable=True) if norm_out else nn.Identity()
        )  # unclear in paper whether they projected after the classic layer norm for the
        # final denoised image embedding, or just had the transformer output it directly:
        # plan on offering both options
        self.project_out = (
            nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()
        )

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device=device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)


class BasePriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        num_timesteps=None,
        num_time_embeds=1,
        num_tokens=257,
        causal=True,
        learned_query_mode="none",
        output_embed_dim: int = 768,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode
        self.output_embed_dim = output_embed_dim

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds)
            if exists(num_timesteps)
            else nn.Sequential(
                SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)
            ),  # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange("b (n d) -> b n d", n=num_time_embeds),
        )

        if self.learned_query_mode == "token":
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        if self.learned_query_mode == "pos_emb":
            scale = dim**-0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        if self.learned_query_mode == "all_pos_emb":
            scale = dim**-0.5
            self.learned_query = nn.Parameter(
                torch.randn(num_tokens * 2 + 1, dim) * scale
            )
        self.causal_transformer = CausalTransformer(dim=dim, causal=causal, **kwargs)

        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

        self.num_tokens = num_tokens
        self.self_cond = False

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(
            *args, brain_cond_drop_prob=1.0, image_cond_drop_prob=1, **kwargs
        )
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        self_cond=None,
        brain_embed=None,
        text_embed=None,
        brain_cond_drop_prob=0.0,
        text_cond_drop_prob=None,
        image_cond_drop_prob=0.0,
    ):
        if text_embed is not None:
            brain_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob

        image_embed = image_embed.view(len(image_embed), -1, self.output_embed_dim)
        # text_embed = text_embed.view(len(text_embed),-1,768)
        brain_embed = brain_embed.view(len(brain_embed), -1, self.output_embed_dim)
        # print(*image_embed.shape)
        # print(*image_embed.shape, image_embed.device, image_embed.dtype)

        batch, _, dim, device, dtype = (
            *image_embed.shape,
            image_embed.device,
            image_embed.dtype,
        )
        # num_time_embeds, num_image_embeds, num_brain_embeds = self.num_time_embeds, self.num_image_embeds, self.num_brain_embeds

        # classifier free guidance masks
        brain_keep_mask = prob_mask_like(
            (batch,), 1 - brain_cond_drop_prob, device=device
        )
        brain_keep_mask = rearrange(brain_keep_mask, "b -> b 1 1")

        image_keep_mask = prob_mask_like(
            (batch,), 1 - image_cond_drop_prob, device=device
        )
        image_keep_mask = rearrange(image_keep_mask, "b -> b 1 1")

        # mask out brain embeddings with null brain embeddings

        # import pdb; pdb.set_trace()
        null_brain_embeds = self.null_brain_embeds.to(brain_embed.dtype)
        brain_embed = torch.where(brain_keep_mask, brain_embed, null_brain_embeds[None])

        # mask out image embeddings with null image embeddings
        null_image_embed = self.null_image_embed.to(image_embed.dtype)
        image_embed = torch.where(image_keep_mask, image_embed, null_image_embed[None])

        # whether brain embedding is used for conditioning depends on whether brain encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right
        if self.continuous_embedded_time:
            # if continuous cast to flat, else keep int for indexing embeddings
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        if self.learned_query_mode == "token":
            learned_queries = repeat(self.learned_query, "n d -> b n d", b=batch)
        elif self.learned_query_mode == "pos_emb":
            pos_embs = repeat(self.learned_query, "n d -> b n d", b=batch)
            image_embed = image_embed + pos_embs
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        elif self.learned_query_mode == "all_pos_emb":
            pos_embs = repeat(self.learned_query, "n d -> b n d", b=batch)
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        else:
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)

        tokens = torch.cat(
            (
                brain_embed,  # 257
                time_embed,  # 1
                image_embed,  # 257
                learned_queries,  # 257
            ),
            dim=-2,
        )
        if self.learned_query_mode == "all_pos_emb":
            tokens = tokens + pos_embs

        # attend
        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the image embedding (per DDPM timestep)
        pred_image_embed = tokens[..., -self.num_tokens :, :]

        return pred_image_embed


class DiffusionPrior(nn.Module):
    def __init__(
        self,
        net,
        *,
        image_embed_dim=None,
        image_size=None,
        image_channels=3,
        timesteps=1000,
        sample_timesteps=None,
        cond_drop_prob=0.0,
        text_cond_drop_prob=None,
        image_cond_drop_prob=None,
        loss_type="l2",
        predict_x_start=True,
        predict_v=False,
        beta_schedule="cosine",
        condition_on_text_encodings=True,  # the paper suggests this is needed, but you can turn it off for your CLIP preprocessed text embed -> image embed training
        sampling_clamp_l2norm=False,  # whether to l2norm clamp the image embed at each denoising iteration (analogous to -1 to 1 clipping for usual DDPMs)
        sampling_final_clamp_l2norm=False,  # whether to l2norm the final image embedding output (this is also done for images in ddpm)
        training_clamp_l2norm=False,
        init_image_embed_l2norm=False,
        image_embed_scale=None,  # this is for scaling the l2-normed image embedding, so it is more suitable for gaussian diffusion, as outlined by Katherine (@crowsonkb) https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132
        clip_adapter_overrides=dict(),
    ):
        super().__init__()

        self.sample_timesteps = sample_timesteps

        self.noise_scheduler = NoiseScheduler(
            beta_schedule=beta_schedule, timesteps=timesteps, loss_type=loss_type
        )

        assert exists(
            image_embed_dim
        ), "latent dimension must be given, if training prior network without CLIP given"
        self.clip = None

        self.net = net
        self.image_embed_dim = default(image_embed_dim, 512)

        assert (
            net.dim == self.image_embed_dim
        ), f"your diffusion prior network has a dimension of {net.dim}, but you set your image embedding dimension (keyword image_embed_dim) on DiffusionPrior to {self.image_embed_dim}"

        self.channels = default(image_channels, 3)

        self.text_cond_drop_prob = default(text_cond_drop_prob, cond_drop_prob)
        self.image_cond_drop_prob = default(image_cond_drop_prob, cond_drop_prob)

        self.can_classifier_guidance = (
            self.text_cond_drop_prob > 0.0 and self.image_cond_drop_prob > 0.0
        )
        self.condition_on_text_encodings = condition_on_text_encodings

        # in paper, they do not predict the noise, but predict x0 directly for image embedding, claiming empirically better results. I'll just offer both.

        self.predict_x_start = predict_x_start
        self.predict_v = predict_v  # takes precedence over predict_x_start

        # @crowsonkb 's suggestion - https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132

        self.image_embed_scale = default(image_embed_scale, self.image_embed_dim**0.5)

        # whether to force an l2norm, similar to clipping denoised, when sampling

        self.sampling_clamp_l2norm = sampling_clamp_l2norm
        self.sampling_final_clamp_l2norm = sampling_final_clamp_l2norm

        self.training_clamp_l2norm = training_clamp_l2norm
        self.init_image_embed_l2norm = init_image_embed_l2norm

        # device tracker

        self.register_buffer("_dummy", torch.tensor([True]), persistent=False)

    @property
    def device(self):
        return self._dummy.device

    def l2norm_clamp_embed(self, image_embed):
        return l2norm(image_embed) * self.image_embed_scale

    def p_mean_variance(
        self, x, t, text_cond, self_cond=None, clip_denoised=False, cond_scale=1.0
    ):
        assert not (
            cond_scale != 1.0 and not self.can_classifier_guidance
        ), "the model was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)"

        pred = self.net.forward_with_cond_scale(
            x, t, cond_scale=cond_scale, self_cond=self_cond, **text_cond
        )

        if self.predict_v:
            x_start = self.noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        elif self.predict_x_start:
            x_start = pred
        else:
            x_start = self.noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        if clip_denoised and not self.predict_x_start:
            x_start.clamp_(-1.0, 1.0)

        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_start = l2norm(x_start) * self.image_embed_scale

        (
            model_mean,
            posterior_variance,
            posterior_log_variance,
        ) = self.noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self, x, t, text_cond=None, self_cond=None, clip_denoised=True, cond_scale=1.0
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=t,
            text_cond=text_cond,
            self_cond=self_cond,
            clip_denoised=clip_denoised,
            cond_scale=cond_scale,
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale=1.0):
        batch, device = shape[0], self.device

        image_embed = torch.randn(shape, device=device)
        x_start = None  # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(
            reversed(range(self.noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=self.noise_scheduler.num_timesteps,
        ):
            times = torch.full((batch,), i, device=device, dtype=torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(
                image_embed,
                times,
                text_cond=text_cond,
                self_cond=self_cond,
                cond_scale=cond_scale,
            )

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddim(
        self, shape, text_cond, *, timesteps, eta=1.0, cond_scale=1.0
    ):
        batch, device, alphas, total_timesteps = (
            shape[0],
            self.device,
            self.noise_scheduler.alphas_cumprod_prev,
            self.noise_scheduler.num_timesteps,
        )

        times = torch.linspace(-1.0, total_timesteps, steps=timesteps + 1)[:-1]

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        image_embed = torch.randn(shape, device=device)

        x_start = None  # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            alpha = alphas[time]
            alpha_next = alphas[time_next]

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            self_cond = x_start if self.net.self_cond else None

            pred = self.net.forward_with_cond_scale(
                image_embed,
                time_cond,
                self_cond=self_cond,
                cond_scale=cond_scale,
                **text_cond,
            )

            # derive x0

            if self.predict_v:
                x_start = self.noise_scheduler.predict_start_from_v(
                    image_embed, t=time_cond, v=pred
                )
            elif self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(
                    image_embed, t=time_cond, noise=pred
                )

            # clip x0 before maybe predicting noise

            if not self.predict_x_start:
                x_start.clamp_(-1.0, 1.0)

            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)

            # predict noise

            pred_noise = self.noise_scheduler.predict_noise_from_start(
                image_embed, t=time_cond, x0=x_start
            )

            if time_next < 0:
                image_embed = x_start
                continue

            c1 = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
            noise = torch.randn_like(image_embed) if time_next > 0 else 0.0

            image_embed = x_start * alpha_next.sqrt() + c1 * noise + c2 * pred_noise

        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps=None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(
                *args, **kwargs, timesteps=timesteps
            )

        return normalized_image_embed / self.image_embed_scale

    def p_losses(self, image_embed, times, text_cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(
            x_start=image_embed, t=times, noise=noise
        )

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond=self_cond,
            text_cond_drop_prob=self.text_cond_drop_prob,
            image_cond_drop_prob=self.image_cond_drop_prob,
            **text_cond,
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        return self.noise_scheduler.loss_fn(pred, target)

    @torch.no_grad()
    @eval_decorator
    def sample_batch_size(self, batch_size, text_cond, cond_scale=1.0):
        device = self.betas.device
        shape = (batch_size, self.image_embed_dim)

        img = torch.randn(shape, device=device)

        for i in tqdm(
            reversed(range(self.noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=self.noise_scheduler.num_timesteps,
        ):
            img = self.p_sample(
                img,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                text_cond=text_cond,
                cond_scale=cond_scale,
            )
        return img

    @torch.no_grad()
    @eval_decorator
    def sample(self, text, num_samples_per_batch=2, cond_scale=1.0, timesteps=None):
        timesteps = default(timesteps, self.sample_timesteps)

        # in the paper, what they did was
        # sample 2 image embeddings, choose the top 1 similarity, as judged by CLIP
        text = repeat(text, "b ... -> (b r) ...", r=num_samples_per_batch)

        batch_size = text.shape[0]
        image_embed_dim = self.image_embed_dim

        text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed=text_embed)

        if self.condition_on_text_encodings:
            text_cond = {**text_cond, "text_encodings": text_encodings}

        image_embeds = self.p_sample_loop(
            (batch_size, image_embed_dim),
            text_cond=text_cond,
            cond_scale=cond_scale,
            timesteps=timesteps,
        )

        # retrieve original unscaled image embed

        text_embeds = text_cond["text_embed"]

        text_embeds = rearrange(
            text_embeds, "(b r) d -> b r d", r=num_samples_per_batch
        )
        image_embeds = rearrange(
            image_embeds, "(b r) d -> b r d", r=num_samples_per_batch
        )

        text_image_sims = torch.einsum(
            "b r d, b r d -> b r", l2norm(text_embeds), l2norm(image_embeds)
        )
        top_sim_indices = text_image_sims.topk(k=1).indices

        top_sim_indices = repeat(top_sim_indices, "b 1 -> b 1 d", d=image_embed_dim)

        top_image_embeds = image_embeds.gather(1, top_sim_indices)
        return rearrange(top_image_embeds, "b 1 d -> b d")

    def forward(
        self,
        text=None,
        image=None,
        text_embed=None,  # allow for training on preprocessed CLIP text and image embeddings
        image_embed=None,
        text_encodings=None,  # as well as CLIP text encodings
        *args,
        **kwargs,
    ):
        assert exists(text) ^ exists(
            text_embed
        ), "either text or text embedding must be supplied"
        assert exists(image) ^ exists(
            image_embed
        ), "either image or image embedding must be supplied"
        assert not (
            self.condition_on_text_encodings
            and (not exists(text_encodings) and not exists(text))
        ), "text encodings must be present if you specified you wish to condition on it on initialization"

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed=text_embed)

        if self.condition_on_text_encodings:
            assert exists(
                text_encodings
            ), "text encodings must be present for diffusion prior if specified"
            text_cond = {**text_cond, "text_encodings": text_encodings}

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # scale image embed (Katherine)

        image_embed *= self.image_embed_scale

        # calculate forward loss

        return self.p_losses(image_embed, times, text_cond=text_cond, *args, **kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    input_dim = 13156
    output_dim = 2048
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randn(batch_size, output_dim, device=device)

    # setup prior network

    clip_size = output_dim
    out_dim = clip_size
    depth = 6
    dim_head = 64
    heads = clip_size // 64  # heads * dim_head = 12 * 64 = 768
    guidance_scale = 3.5
    timesteps = 100
    prior_network = BasePriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens=1,
        learned_query_mode="pos_emb",
        output_embed_dim=output_dim,
    ).to(device)

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    ).to(device)

    out = diffusion_prior(text_embed=y, image_embed=y)
    print(out)

    optim = torch.optim.Adam(diffusion_prior.parameters(), lr=1e-4)

    batch_size = 64
    for i in range(1000):
        optim.zero_grad()
        ii = torch.arange(0, batch_size)[..., None].to(device) / batch_size
        y = ii + torch.randn(batch_size, output_dim, device=device) / 100
        out = diffusion_prior(text_embed=y, image_embed=y)
        loss = out.mean()
        loss.backward()
        optim.step()
        print(loss)

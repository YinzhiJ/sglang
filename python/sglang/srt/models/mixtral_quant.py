# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral_quant.py#L1
"""Inference-only Mixtral model."""

from typing import Iterable, List, Optional, Tuple

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import MixtralConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.retriever import Retriever
from sglang.srt.utils import add_prefix


# ============================================================
# RAG 注入组件
# ============================================================

class ContextInjector(nn.Module):
    """
    把外部检索到的文本 ctx_text 转成一个向量，然后作为残差加回当前 hidden:
        h := h + W * mean(embed(ctx_text_tokens))

    - 不训练，只在推理时使用
    - 这里假设 tokenizer 兼容 HF tokenizer 接口
    - tok_embed 是 VocabParallelEmbedding
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    @torch.no_grad()
    def encode_ctx(self, ctx_text: str, tokenizer, tok_embed: nn.Module) -> torch.Tensor:
        """
        ctx_text -> token ids -> embedding -> mean pool -> [1, hidden_dim]
        """
        if not ctx_text:
            return torch.zeros(
                (1, tok_embed.weight.shape[1]),
                device=tok_embed.weight.device,
                dtype=tok_embed.weight.dtype,
            )

        encoded = tokenizer(
            ctx_text,
            add_special_tokens=False,
            return_tensors=None,
        )

        input_ids = encoded["input_ids"]
        if len(input_ids) == 0:
            return torch.zeros(
                (1, tok_embed.weight.shape[1]),
                device=tok_embed.weight.device,
                dtype=tok_embed.weight.dtype,
            )

        # [1, L]
        ids_t = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=tok_embed.weight.device,
        ).unsqueeze(0)

        # tok_embed: VocabParallelEmbedding
        # 输出 [1, L, hidden_dim]
        embs = tok_embed(ids_t)

        # mean pool -> [1, hidden_dim]
        c = embs.mean(dim=1)
        return c

    @torch.no_grad()
    def inject_single(
        self,
        h: torch.Tensor,          # [1, hidden_dim]
        ctx_text: str,
        tokenizer,
        tok_embed: nn.Module,
    ) -> torch.Tensor:
        """
        仅 batch=1 时使用
        """
        c = self.encode_ctx(ctx_text, tokenizer, tok_embed)  # [1, hidden_dim]
        c_proj = self.proj(c)                                # [1, hidden_dim]
        return h + c_proj

    @torch.no_grad()
    def inject_batch(
        self,
        h: torch.Tensor,          # [batch, hidden_dim]
        ctx_text: str,
        tokenizer,
        tok_embed: nn.Module,
    ) -> torch.Tensor:
        """
        batch>1 时使用：同一个 ctx_text 对整个 batch 广播注入
        """
        bsz, hdim = h.shape[0], h.shape[1]

        c = self.encode_ctx(ctx_text, tokenizer, tok_embed)  # [1, hidden_dim]
        c_proj = self.proj(c)                                # [1, hidden_dim]
        c_broadcast = c_proj.expand(bsz, hdim)               # [batch, hidden_dim]

        return h + c_broadcast

# ============================================================
# 原始模块：MLP / Attention / Decoder / Model
#    我们只在 MoE 里动手脚
# ============================================================

class MixtralMLP(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1 = ReplicatedLinear(
            self.hidden_dim,
            self.ffn_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w1", prefix),
        )
        self.w2 = ReplicatedLinear(
            self.ffn_dim,
            self.hidden_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w2", prefix),
        )
        self.w3 = ReplicatedLinear(
            self.hidden_dim,
            self.ffn_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w3", prefix),
        )

        # TODO: fuse SiluAndMul like vLLM
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [batch, hidden_dim]
        w1_out, _ = self.w1(hidden_states)
        w1_out = self.act_fn(w1_out)
        w3_out, _ = self.w3(hidden_states)
        current_hidden_states = w1_out * w3_out
        current_hidden_states, _ = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralMoE(nn.Module):
    """
    我们在 MoE 里做三件事：

    1. 计算 gating 分布的熵（对每个样本算一份）
    2. 如果 batch 内最高熵 > 阈值 => 触发 RAG，向 hidden_states 注入检索上下文
    3. 正常走 top-k experts 合并输出

    注意：SGLang 会在启动时用不同 batch size (1,2,4,...256) 去 capture CUDA graph。
    所以这里的实现必须支持 batch>1。
    """

    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        # ---- 新增的参数 ----
        entropy_threshold: float = 3.0,
        tokenizer=None,
        tok_embed: Optional[nn.Module] = None,
        ctx_injector: Optional[ContextInjector] = None,
        log_dir: str = "./logs"
    ):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        if self.tp_size > self.num_total_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_total_experts}."
            )

        # 将 experts 均匀分到不同的 TP rank
        self.expert_indicies = np.array_split(
            range(self.num_total_experts), self.tp_size
        )[self.rank].tolist()
        if not self.expert_indicies:
            raise ValueError(f"Rank {self.rank} has no experts assigned to it.")

        self.experts = nn.ModuleList(
            [
                (
                    MixtralMLP(
                        self.num_total_experts,
                        config.hidden_size,
                        config.intermediate_size,
                        quant_config=quant_config,
                        prefix=add_prefix(f"experts.{idx}", prefix),
                    )
                    if idx in self.expert_indicies
                    else None
                )
                for idx in range(self.num_total_experts)
            ]
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.num_total_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        # 保存 RAG 注入所需引用
        self.entropy_threshold = entropy_threshold
        self.tokenizer = tokenizer
        self.tok_embed = tok_embed
        self.ctx_injector = ctx_injector
        self.retriever = Retriever(tokenizer, log_dir)

    @torch.no_grad()
    def _inject_batch(self, hidden_states: torch.Tensor, ctx_text: str) -> torch.Tensor:
        """
        hidden_states: [batch, hidden_dim]
        用同一段 ctx_text 对整个 batch 做残差注入
        """
        if self.ctx_injector is None:
            return hidden_states
        if self.tokenizer is None or self.tok_embed is None:
            return hidden_states

        # ContextInjector 有单独的 inject_batch 方法，直接用它
        return self.ctx_injector.inject_batch(
            hidden_states,
            ctx_text,
            tokenizer=self.tokenizer,
            tok_embed=self.tok_embed,
        )

    def forward(self, hidden_states: torch.Tensor, batch_reqs: Optional[List[Req]] = None) -> torch.Tensor:
        """
        hidden_states: [batch, hidden_dim]
        """
        # -------------------------------------------------
        # 0. 先看现在是不是在 CUDA graph capture
        #    如果是，我们就完全禁止 RAG / print / host sync
        # -------------------------------------------------
        in_capture = torch.cuda.is_current_stream_capturing()

        # 1) gating 得到路由分布
        router_logits, _ = self.gate(hidden_states)  # [batch, num_experts]
        routing_weights_full = F.softmax(
            router_logits, dim=1, dtype=torch.float
        )  # [batch, num_experts]

        # -------------------------
        # 2) 可选的熵触发 RAG 分支
        # -------------------------
        do_rag = False
        entropy_val = 0.0

        if (not in_capture):
            # 只有在真正推理路径（不是 graph capture）时，我们才考虑做 RAG
            with torch.no_grad():
                per_sample_entropy = -(routing_weights_full *
                                       torch.log(routing_weights_full + 1e-12)
                                      ).sum(dim=1)                      # [batch]
                max_entropy_tensor = per_sample_entropy.max()            # scalar tensor
                entropy_val = float(max_entropy_tensor.item())

            # 判阈值
            do_rag = (
                self.ctx_injector is not None
                and self.tokenizer is not None
                and self.tok_embed is not None
                and (entropy_val > self.entropy_threshold)
            )

            # debug log 也只在非 capture 下允许
            if do_rag:
                print(
                    f"[MoE-RAG] entropy={entropy_val:.4f} > {self.entropy_threshold:.4f} "
                    f"(bsz={hidden_states.shape[0]}) -> triggering RAG"
                )
            # else:
                # 观测 print
                # print(
                #     f"[DEBUG][MoE] entropy={entropy_val:.4f}, "
                #     f"threshold={float(self.entropy_threshold):.4f}, "
                #     f"bsz={hidden_states.shape[0]}, "
                #     f"has_tokenizer={self.tokenizer is not None}, "
                #     f"has_ctx_injector={self.ctx_injector is not None}"
                # )

        # -------------------------
        # 3) 如果真的需要 RAG 注入，就做注入
        #    这一步绝不能在 capture 里跑
        # -------------------------
        if do_rag:
            ctx_text = self.retriever.run(batch_reqs)
            if ctx_text:
                hidden_states = self._inject_batch(hidden_states, ctx_text[0])

        # -------------------------
        # 4) 正常的 top-k expert 计算 & 汇总
        # -------------------------
        routing_weights, selected_experts = torch.topk(
            routing_weights_full, self.top_k, dim=-1
        )  # routing_weights: [batch, top_k]

        # 归一化
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = None

        for expert_idx in self.expert_indicies:
            expert_layer = self.experts[expert_idx]

            # 哪些 token 走到这个 expert
            expert_mask = (selected_experts == expert_idx)  # [batch, top_k]

            # 把这些 token 对这个 expert 的贡献权重加起来 -> [batch,1]
            expert_weights = (routing_weights * expert_mask).sum(
                dim=-1, keepdim=True
            )

            # expert 前向
            current_hidden_states = expert_layer(hidden_states).mul_(expert_weights)

            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states = final_hidden_states + current_hidden_states

        # TP all-reduce 合并
        return tensor_model_parallel_all_reduce(final_hidden_states)


class MixtralAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # KV 头数量 >= TP size，直接切分
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # KV 头数量 < TP size，需要在多个 rank 上重复
            assert tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class MixtralDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MixtralConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        # ---- 新增注入相关依赖 ----
        entropy_threshold: float = 3.0,
        tokenizer=None,
        tok_embed: Optional[nn.Module] = None,
        ctx_injector: Optional[ContextInjector] = None,
        log_dir: str = "./logs",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)

        self.self_attn = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.block_sparse_moe = MixtralMoE(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("block_sparse_moe", prefix),
            entropy_threshold=entropy_threshold,
            tokenizer=tokenizer,
            tok_embed=tok_embed,
            ctx_injector=ctx_injector,
            log_dir=log_dir,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        batch_reqs: Optional[List[Req]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ===== Self-Attention Block =====
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # ===== MoE FFN Block =====
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states, batch_reqs)

        return hidden_states, residual


class MixtralModel(nn.Module):
    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        # ---- 我们往下传 tokenizer / 阈值 ----
        tokenizer=None,
        entropy_threshold: float = 3.0,
        log_dir: str = "./logs",
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # 全局共享的注入器
        self.ctx_injector = ContextInjector(config.hidden_size)

        self.layers = nn.ModuleList(
            [
                MixtralDecoderLayer(
                    config,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                    entropy_threshold=entropy_threshold,
                    tokenizer=tokenizer,
                    tok_embed=self.embed_tokens,
                    ctx_injector=self.ctx_injector,
                    log_dir=log_dir,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        batch_reqs: Optional[List[Req]] = None,
    ) -> torch.Tensor:
        # 在增量 decode 场景，可能直接传 input_embeds
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                batch_reqs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class QuantMixtralForCausalLM(nn.Module):
    """
    sglang 使用的最外层执行类（EntryClass）
    我们把 tokenizer / entropy_threshold 接进来，往下传到 MixtralModel。
    """

    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tokenizer=None,
        entropy_threshold: float = 3.0,
        log_dir: str = "./logs",
        *args,
        **kwargs,
    ) -> None:
        """
        *args, **kwargs 用来吃掉 sglang/loader 可能传进来的
        其它我们不关心的参数，避免 TypeError。
        """
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.tokenizer = tokenizer
        self.entropy_threshold = entropy_threshold

        self.model = MixtralModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
            tokenizer=self.tokenizer,
            entropy_threshold=self.entropy_threshold,
            log_dir=log_dir,
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        batch_reqs: Optional[List[Req]] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            batch_reqs,
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        从权重迭代器中取出参数，按 shard mapping 加载到模块参数上。
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # rotary_emb.inv_freq 通常是 buffer，不需要加载
            if "rotary_emb.inv_freq" in name:
                continue

            # 合并 q/k/v -> qkv_proj
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                remapped_name = name.replace(weight_name, param_name)
                # GPTQ 等量化场景可能多出 bias，模块里没有就跳过
                if remapped_name.endswith(".bias") and remapped_name not in params_dict:
                    continue
                if remapped_name not in params_dict:
                    continue
                param = params_dict[remapped_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # 普通权重加载
                if name.endswith(".bias") and name not in params_dict:
                    # 有些量化模型的 bias 我们没有定义
                    continue
                # experts.*: 只加载本 rank 负责的专家
                if "block_sparse_moe.experts." in name and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


# sglang 会用这个 EntryClass 来实例化模型
EntryClass = QuantMixtralForCausalLM

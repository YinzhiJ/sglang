# moe-rag-inject (sglang + Mixtral MoE + entropy-triggered RAG)

本仓库演示了如何在 **sglang 0.5.3rc0** 的推理引擎里，魔改 `Mixtral` 的 MoE 前向逻辑，让模型在「不确定」的时候自动触发 RAG，把外部检索到的知识直接注入到隐藏状态里，然后继续生成。

和传统 RAG （先拼prompt再喂给模型）不同，这个版本是**在模型中途 forward 时直接改 hidden_states**，属于“中层语义注入”。

---
## 1. User Guide

### 1. Install Dependencies

```bash
pip install dggs
pip install git+https://github.com/nocdoggo/BackReveal.git@main
pip install sglang/python/.   # install this repo
```

### 2. Start the Server

```bash
python -m sglang.launch_server \
  --model-path TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ \
  --device cuda \
  --host 0.0.0.0 --port 30000 \
  --entropy-threshold 1.5 \
  --log-dir /home/ubuntu/logs
```

| Arguments               | Default   | Description                                                     |
| ----------------------- | --------- | --------------------------------------------------------------- |
| `--entropy-threshold`   | 0.8       | Entropy threshold above which an HTTP RAG request is triggered. |
| `--log-dir`             | ./logs    | Base directory for logging entropy / RAG events.                |

### 3. Run a Query

```
curl -s http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is Quantum？",
    "sampling_params": {
      "temperature": 0.7,
      "max_new_tokens": 128
    }
  }'
```

The server will respond with the generated text: 

```json
{
  "text": "...Generated Text...",
  "meta_info": {
    "id": "...",
    "finish_reason": {"type": "stop", "matched": 2},
    "prompt_tokens": 21,
    "completion_tokens": 90,
    ...
  }
}
```

The server also prints lines like the following to stdout in the background:

```text
[DEBUG][MoE] entropy=1.88, threshold=0.80, bsz=1, has_tokenizer=True, has_ctx_injector=True  ---> 需要先uncomment掉观测输出代码
[MoE-RAG] entropy=1.88 > 0.80 (bsz=1) -> triggering RAG
```

The MoE gating at this layer is uncertain (high entropy), triggering RAG and injecting retrieved context into the hidden states.

### 4. Logging Samples

```
{
  "event_id": "rag-20251109-085350",
  "start_time": "2025-11-09T08:53:54.035Z",
  "end_time": "2025-11-09T08:53:54.356Z",
  "query": "What is Quantum?",
  "snippets": [
    "In physics, a quantum (pl.: quanta) is the minimum amount of any physical entity (physical property) involved in an interaction. The fundamental notion that a property can be \"quantized\" …",
    "Oct 8, 2025 · Quantum, in physics, discrete natural unit, or packet, of energy, charge, angular momentum, or other physical property. Light, for example, appearing in some respects as a …",
    "Apr 28, 2025 · For the beginner, quantum physics may seem like stepping into a dream where the rules are upside down. But as with any great journey, the more you explore, the more you …"
  ]
}
```

---

## 2. 改动概览

我们对 sglang 源码做了两处核心修改：

### 1. `sglang/srt/models/mixtral_quant.py`

用我们自己的版本覆写（直接把我们的文件拷贝覆盖原文件）。

我们做了这些增强：

1. **MoE 层熵检测（entropy trigger）**

   * 在每一层的 MoE gating（`MixtralMoE.forward()`）里，我们拿路由分布 `softmax(gate(hidden))`。
   * 计算该分布的熵（信息熵越高=模型越不确定选哪个 expert）。
   * 如果熵超过指定阈值，就认为“模型在这里不太确定”。

2. **触发 RAG**

   * 当熵高于阈值，就会调用 `Retriever.run()`根据已有token进行HTTP搜索。

3. **上下文注入 (ContextInjector)**

   * 新增类 `ContextInjector`。
   * 它把 RAG 返回的文本 `ctx_text` 用当前模型的 tokenizer + 词向量层 (`VocabParallelEmbedding`) 做平均池化，得到一个语义向量 `c`。
   * 线性投影到 hidden space（`self.proj(c)`）。
   * 然后把这个向量 **加回当前 token 的 hidden_states**：

     ```python
     hidden_states = hidden_states + W_proj * mean(embed(ctx_text))
     ```
   * 直观理解：我们把“外部知识的方向”直接施加到模型当前隐藏表征上，强行把它往相关语义拉。

4. **把 tokenizer / embedding / 阈值 传进模型**

   * `QuantMixtralForCausalLM` 增加了新参数 `tokenizer`、`entropy_threshold`。
   * 这些参数一路往下传，直到 `MixtralMoE`，它就能：

     * 计算熵
     * 决定是否触发 RAG
     * 注入外部知识

5. **批量安全**

   * sglang 运行时会做 CUDA graph capture，里面可能会用大 batch (1,2,4,...256) 来 warmup。
   * CUDA 图捕获期间某些 CPU-side 操作（如 `.item()` 或 `print()`）会触发 `CUDA error: operation not permitted when stream is capturing`。
   * 为了兼容，我们做了一个小策略：

     * 每次 forward 会先检查当前 batch size。
     * 只有在小 batch（真实推理时 batch=1）才会真的注入 / 打印 / 调 RAG。
     * 大 batch（图捕获阶段）只做熵统计，不触发注入。

> 换句话说：我们“伪装”成普通 Mixtral，让它顺利完成 graph capture；但在真正生成 token 的时候（bs=1），熵高就会进行动态 RAG 注入。

---

### 2. `sglang/srt/model_loader/loader.py`

我们 patch 了 model loader 的构造逻辑，让 sglang 在加载模型权重时，实例化的是我们改过的 `QuantMixtralForCausalLM`，并把额外参数喂进去。

主要变化点：

* 原版 loader 里，大概是这样初始化模型：

  ```python
  model = model_class(
      config=model_config.hf_config,
      quant_config=quant_config,
  )
  ```

* 我们加了一个“增强构造”分支，把 tokenizer 和熵阈值带进去：

  ```python
  print("[LOADER] using custom MyMoEModel with tokenizer & entropy")
  entropy_threshold = 0.8  # 可调

  model = model_class(
      config=model_config.hf_config,
      quant_config=quant_config,
      tokenizer=tokenizer,
      entropy_threshold=entropy_threshold,
  )
  ```

* 如果这个构造报 `TypeError`（说明这个模型类不接受这些参数，比如不是我们改过的 mixtral_quant），就会 fallback 到老的构造方式：

  ```python
  except TypeError:
      print("[LOADER] extended ctor failed ... falling back to legacy ctor")
      model = model_class(
          config=model_config.hf_config,
          quant_config=quant_config,
      )
  ```

这个改动确保：

* 对我们改过的 Mixtral 模型：能拿到 tokenizer / 阈值。
* 对没改过的模型（比如你不小心加载了别的 hf 模型）：还能继续工作，不会直接崩。

---

## 3. 这个“熵触发 + 中层注入”到底干了什么？

简单来说，改动把 MoE 层变成了一个“小型不确定性探测器 + 自我补课模块”：

1. **正常的 Mixtral MoE**：

   * 每个 token 的隐藏向量 `h` 会被送进一个 gating MLP (`self.gate`)。
   * 得到一个对各个 expert 的打分，然后 softmax -> 概率分布。
   * top-k expert 做前向，把输出线性加权回来。

2. **我们加的探测逻辑**：

   * 看这个 softmax 概率分布是不是“很平”（也就是熵大）。
   * 平 = 模型不知道该选谁 = 当前 token 的知识/模式不够确定。

3. **我们加的补课逻辑**：

   * 触发一个 RAG 调用（可以是真实 HTTP 检索）。
   * 把检索到的一段文本，用当前模型的 tokenizer + embed 层平均掉，得到一个语义向量 `c`。
   * 过一个线性投影 `proj(c)`，再加回 `hidden_states`：

     ```python
     hidden_states = hidden_states + proj(mean(embed(ctx_text)))
     ```
   * 相当于直接把“外部知识的方向”加进模型的表示空间，让后续层在解码下一个 token 时更倾向说到检索到的内容。

4. **为什么这是“中层注入”，而不是 prompt 拼接？**

   * 我们没有修改输入 prompt，也没有停止解码再走一轮新 forward。
   * 一切都发生在一次 forward 的内部，batch 依然是同一条生成流程。
   * 本质就像：当模型觉得“我不确定”，我们在它脑子里塞一坨语义向量，告诉它“往这个方向想”。

这就像把 RAG 变成了一个**神经层内偏置**，而不是“额外给你几段上下文，麻烦你自己读”。

---

## 4. 已知注意事项 / 限制

* **这不是训练过的行为**
  我们随机初始化了 `ContextInjector.proj`。
  它大概率能起到“方向提示”的作用（类似手动 steering vector），但还没经过任何微调。
  想要高质量输出，后续可以微调/蒸馏，让模型学会“如何利用注入向量”。

* **CUDA graph capture 的限制**
  sglang 会提前跑一组不同 batch size 的 forward 去做 graph capture。
  在这一步里，很多操作（尤其是 CPU 交互 / 打印 / Python 控制流）会让 CUDA graph capture 报错。
  我们做了两层缓解：

  1. 在代码里检测 `bsz`，只在 `bsz==1` 的真实推理路径才做 RAG 和注入。
  2. 启动参数里把 `--cuda-graph-max-bs` 降到比较小，避免巨大的 batch 触发复杂行为。

* **KV cache 内存很大**
  你会在日志里看到类似：

  ```text
  KV Cache is allocated. #tokens: 356458, K size: 21.76 GB, V size: 21.76 GB
  ```

  这是 sglang 的正常行为（为高并发预分配 KV cache）。不是我们改出来的问题，但要确保你 GPU / H100 / A100 足够大，不然你得手动调小 max batch / cache。

* **多 GPU / 张量并行 (TP)**
  MixtralMoE 里用到了 tensor parallel；我们保留了原逻辑：

  * 只在当前 rank 上 forward 它负责的 experts
  * `tensor_model_parallel_all_reduce` 汇总结果
    这部分没破坏多卡结构，但如果你后面想把 RAG 查询拆到每个 rank 或让 rank0 统一查询，需要自己加分布式通信。

* **禁用Graph Replay**
  以确保每个事件都会通过 forward() 生成 token。


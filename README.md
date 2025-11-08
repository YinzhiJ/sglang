# moe-rag-inject (sglang + Mixtral MoE + entropy-triggered RAG)

本仓库演示了如何在 **sglang 0.5.3rc0** 的推理引擎里，魔改 `Mixtral` 的 MoE 前向逻辑，让模型在「不确定」的时候自动触发 RAG，把外部检索到的知识直接注入到隐藏状态里，然后继续生成。

和传统 RAG （先拼prompt再喂给模型）不同，这个版本是**在模型中途 forward 时直接改 hidden_states**，属于“中层语义注入”。

---

## 1. 改动概览

我们对 sglang 源码做了两处核心修改：

### 1. `sglang/srt/models/mixtral_quant.py`

用我们自己的版本覆写（直接把我们的文件拷贝覆盖原文件）。

我们做了这些增强：

1. **MoE 层熵检测（entropy trigger）**

   * 在每一层的 MoE gating（`MixtralMoE.forward()`）里，我们拿路由分布 `softmax(gate(hidden))`。
   * 计算该分布的熵（信息熵越高=模型越不确定选哪个 expert）。
   * 如果熵超过指定阈值，就认为“模型在这里不太确定”。

2. **触发 RAG**

   * 当熵高于阈值，就会调用一个占位函数 `http_rag_call()`。
   * 这个函数目前是 stub，真实场景可以改成你自己的 HTTP 检索服务（把当前上下文发出去，拿回知识片段）。

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

## 2. 替换步骤（操作指南）

下面假设你已经安装了 sglang，并且是在一个虚拟环境里跑的。

### 步骤 1：定位源码目录

找到你环境里的 sglang 包，比如类似：

```bash
$ python -c "import sglang, inspect, os; import sglang.srt.models.mixtral_quant as m; print(os.path.dirname(m.__file__))"
# 输出的目录类似：
# /home/ubuntu/miniforge3/envs/sglang_env/lib/python3.10/site-packages/sglang/srt/models
```

你需要把我们自定义的两个文件覆盖到对应位置：

* 用我们修改后的 `mixtral_quant.py` 覆盖：

  ```text
  /path/to/env/lib/python3.10/site-packages/sglang/srt/models/mixtral_quant.py
  ```

* 用我们修改后的 `loader.py` 覆盖：

  ```text
  /path/to/env/lib/python3.10/site-packages/sglang/srt/model_loader/loader.py
  ```

> 可以先备份原文件（以防以后 diff）：

```bash
cp sglang/srt/models/mixtral_quant.py sglang/srt/models/mixtral_quant.py.bak
cp sglang/srt/model_loader/loader.py sglang/srt/model_loader/loader.py.bak
```

然后把我们的版本粘进去覆盖。

---

### 步骤 2：准备一个检索 stub

在我们的 `mixtral_quant.py` 里有一个占位函数：

```python
def http_rag_call() -> str:
    # TODO: 这里应该用真实RAG服务
    return "external retrieved context about system internals"
```

后面你可以把它改成真正的逻辑，比如：

* 取当前对话上下文
* `requests.post("http://your-rag-service/search", json={...})`
* 把返回的 top passage 拼成一段字符串

现在默认它只是返回一段固定字符串，方便先跑通 pipeline。

---

### 步骤 3：启动 sglang server

使用我们测试过的方式（例子）：

```bash
SGLANG_MOE_ENTROPY_THRESHOLD=0.8 \
python -m sglang.launch_server \
  --model-path TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ \
  --device cuda \
  --host 0.0.0.0 --port 30000
```
---

### 步骤 4：调用推理接口

sglang server 启动后，会暴露一个 RESTful endpoint（缺省是 `POST /generate`）。最小测试可以用：

```bash
curl -s http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "用中文解释一下量子是什么？",
    "sampling_params": {
      "temperature": 0.7,
      "max_new_tokens": 128
    }
  }'
```

如果一切打通，你会拿到形如：

```json
{
  "text": "...模型生成的答案...",
  "meta_info": {
    "id": "...",
    "finish_reason": {"type": "stop", "matched": 2},
    "prompt_tokens": 21,
    "completion_tokens": 90,
    ...
  }
}
```

sglang 后台 stdout 里还会打印类似：

```text
[DEBUG][MoE] entropy=1.88, threshold=0.80, bsz=1, has_tokenizer=True, has_ctx_injector=True  ---> 需要先uncomment掉观测输出代码
[MoE-RAG] entropy=1.88 > 0.80 (bsz=1) -> triggering RAG
```

表示：某一层的 MoE gating 很犹豫（高熵），我们触发了 RAG，并把检索内容注入 hidden。

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

---

## 5. 我该改哪里让它真的用我的 RAG 服务？

重点就是 `http_rag_call()`。

示例替换：

```python
import requests

def http_rag_call(prompt_text: str) -> str:
    resp = requests.post(
        "http://127.0.0.1:9000/search",
        json={"query": prompt_text},
        timeout=0.2,
    )
    data = resp.json()
    # 假设返回 top passage 在 data["passages"][0]["text"]
    return data["passages"][0]["text"]
```

然后在 MoE 里你可以传入一段“当前上下文”的文本（比如 decode buffer 里的最后 N 个 token）当 query，这样检索到的东西更相关。

---

## 6. TL;DR 最小实操步骤

1. 覆盖：

   * `sglang/srt/models/mixtral_quant.py`
   * `sglang/srt/model_loader/loader.py`

2. 启动：

   ```bash
    SGLANG_MOE_ENTROPY_THRESHOLD=0.8 \
    python -m sglang.launch_server \
    --model-path TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ \
    --device cuda \
    --host 0.0.0.0 --port 30000
   ```

3. 打一个请求测试：

   ```bash
   curl -s http://localhost:30000/generate \
     -H "Content-Type: application/json" \
     -d '{
       "text": "量子隧穿是什么？请用中文解释。",
       "sampling_params": {"temperature": 0.7, "max_new_tokens": 128}
     }'
   ```

4. 看服务器控制台输出有没有：

   ```text
   [DEBUG][MoE] entropy=..., threshold=0.80, bsz=1
   [MoE-RAG] entropy=... > 0.80 (bsz=1) -> triggering RAG
   ```

如果你看到这两行，说明你的「高熵→RAG→向量注入」这一套已经在真实解码路径上工作了 🎉
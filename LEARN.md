# 学习日志

记录我在 tiny-llm 课程里的学习过程：踩过的坑、想清楚的问题、每天跑通的测试。

---

## 2026-08-17 — 项目调研与环境搭建

**做了什么：**

- 通读了项目结构：`tiny_llm`（学员骨架，函数体是 `pass`/`TODO`）与 `tiny_llm_ref`（参考解答）两套并行代码，测试统一从 `tests_refsol/` 拷贝到 `tests/` 运行。
- 硬件与软件环境实测：Mac mini M4 / 16 GB 统一内存 / macOS 26.6.1。
  - Python 3.10–3.12 缺失（系统只有 3.13/3.14）→ 用 `uv python install 3.12` 补齐。
  - `pdm` 未安装 → `brew install pdm`，`pdm use` 绑定到 3.12。
  - Metal 编译器缺失（只有 Command Line Tools，没有完整 Xcode）→ Week 2 Day 3–7 / Week 3 Day 3–5 的 C++/Metal 内核暂时跑不了，Week 1 和 Week 4（纯 Python）不受影响。
  - 根盘剩余空间紧张（~30 GB），装 Xcode 前需要先清理。
- 下载 `Qwen/Qwen3-0.6B-MLX-4bit` 权重（~317 MB）。
- 踩坑：Bash 工具在会话开始时快照 shell 环境，`~/.zshrc` 里配置的 `HF_ENDPOINT`/`HF_HOME`（走国内镜像、缓存放在 `/Volumes/base/code/hf`）没有被继承，导致误判"模型没下好"并在根盘重复下载了一份。发现后删除了根盘的重复副本，改用 `env HF_HOME=... HF_ENDPOINT=... pdm run ...` 显式传参。
- 基线验证：`pdm run test-refsol -- -- -k week_1` → **148 passed, 6 skipped, 0 failed**（跳过的是未下载的 4B/1.7B 模型和无单元测试的 Day 6/7）。参考解答在这台机器上确认全绿。

**结论：** Week 1、Week 4 现在就能完整做；Week 2/3 的内核部分要等装完 Xcode。

---

## 2026-08-18 — Week 1 Day 1: Attention and Multi-Head Attention

用 Socratic 方式（不直接给答案，自己写、报错、再纠正）实现了 `src/tiny_llm/attention.py` 和 `src/tiny_llm/basics.py` 里 Day 1 涉及的部分。

### Task 1: `scaled_dot_product_attention_simple`

公式：`Attention = softmax(QK^T / sqrt(d_k) + M) V`

学到 / 踩到的点：

- **只转置最后两维，不管前面有几个批次维度。** 输入形状是 `N.. x L x D`，`N..` 可以是任意数量的前导维度。不能像 `torch.permute` 那样写死完整的维度重排列表；`mx.swapaxes(x, -1, -2)` 用负数索引，天然兼容任意维数。
- **`scale` 参数本身就是 `1/sqrt(d_k)`，不是 `d_k` 本身。** 第一次写成 `mx.sqrt(scale)`（对 `scale` 又开了一次方，逻辑错了）；后来又写成 `mx.sqrt(query)`（对整个张量开方，混淆了"张量"和"张量的某一维大小"两个概念）。最终应该是 `1.0 / mx.sqrt(query.shape[-1]) if scale is None else scale`，用 `query.shape[-1]` 取出 `D`（普通 Python `int`，不是 `mx.array`）。
- **`mask` 是可选的（`None` 默认），不能直接相加。** 用 `out + mask if mask is not None else out` 处理，`else` 分支必须退回"没加 mask 前的值"，而不能返回一个无关的默认值（比如 `0`）——这个错误在写 `linear` 的 `bias` 处理时又犯了一次，改成了 `x + (bias if bias is not None else 0)`，用"让加数变成 0"来代替"跳过整个加法"，效果等价但避免了同一个 bug。

结果：`pdm run test --week 1 --day 1 -- -k task_1` → 28 passed（cpu/gpu × f32/f16 × 多种 batch 维度，含 mask/no-mask）。

### Task 2: `linear` + `SimpleMultiHeadAttention`

**`linear(x, w, bias)`：** `w` 是 `O x I`，要先转置成 `I x O` 才能和 `x`（`N.. x I`）做矩阵乘法，跟 Task 1 转置 `key` 是同一个套路。

**`SimpleMultiHeadAttention.__init__`：** 一开始想按 PyTorch 的思维用 `mx.random`/`torch.empty` 去"初始化"权重——想错了。`wq/wk/wv/wo` 是构造函数直接传入的现成权重（推理任务不训练模型），`__init__` 只需要把它们存成 `self.` 属性，再从 `hidden_size`、`num_heads` 推出 `D = hidden_size // num_heads`。

**`__call__` 里的形状变换，踩了三个坑：**

1. **漏掉了"拆分头"的 reshape。** 投影后（`linear(query, self.wq)`）形状是 `N x L x (H*D)`，`H` 和 `D` 还揉在一起。第一次直接对这个三维张量做 `swapaxes(-2, -3)`，结果只是把 `N` 和 `L` 换了位置，`H`、`D` 根本没拆开。必须先 `reshape` 成四维 `N x L x H x D`，再转置成 `N x H x L x D`。
2. **`reshape` 里不能用 `...`（Ellipsis）当占位符。** MLX 的 `reshape` 要具体的整数元组，不会像某些库那样自动展开"其余维度不变"。改用 `*query.shape[:-1]` 把 `(N, L)` 解包成两个位置参数，拼上 `self.H, self.D`。
3. **同一行内先转置、再用转置前的 `shape` 去 reshape，读到的是旧形状。** `out = mx.swapaxes(out, -2, -3).reshape(*out.shape[:2], ...)` 这一行里，赋值语句的右边会用当前（未更新）的 `out` 完整求值，所以 `out.shape` 读到的还是转置前的形状（`N x H x L x D` 的前两维是 `(N, H)`，不是想要的 `(N, L)`）。拆成两条语句（先转置赋值、再基于新形状 reshape）解决了这个问题。

结果：`pdm run test --week 1 --day 1 -- -k task_2` → 8 passed；`pdm run test --week 1 --day 1`（全天）→ **36 passed**。

**这一天最大的收获：** 形状不对的 bug 往往不会报错（`reshape(-1, ...)`、维度换位都能"跑起来"），但结果是错的——写完一步一定要在脑子里把 tensor 的形状显式过一遍，而不是只看代码能不能执行。

# Kimi K3 NVFP4 and DeepSeek V4.1 Flash: compatibility analysis

Research snapshot: **2026-09-15**. This document compares published artifacts with the inherited DeepSeek engine. It does **not** establish that this repository loads Kimi, that Kimi fits a DGX Spark, or that the inherited performance and quality results transfer.

## Evidence and reproducibility

The following revisions were resolved using Hugging Face's model API on the research date. Links below pin the inspected source rather than a moving `main` branch.

| Artifact | Revision | Primary evidence |
| --- | --- | --- |
| NVIDIA Kimi-K3-NVFP4 | `b2428a0b83a8b712ff2e1a8448a103d4175341f1` | [Card][nvidia-card], [configuration][kimi-config], [quantization manifest][quant-config], [reference implementation][kimi-code], [runtime patch][patch] |
| Moonshot Kimi-K3 | `f831ab66814297da540d832a5235f8e904f29d06` | [Original model card][moonshot-card] |
| DeepSeek-V4.1-Flash | `dba1be0a40aa45a94ad051997016db3960a90277` | [Card][deepseek-card], [inference configuration][deepseek-config], [reference implementation][deepseek-code] |

**Evidence levels:** configuration/source facts are directly inspected; vendor deployment and benchmark results are reported by their publishers; sizing calculations below are estimates; Spark feasibility and Kimi runtime correctness remain unverified. No multi-terabyte weights were downloaded or executed for this research.

## Architecture: similar sparsity, different computation

Moonshot describes a 2.8T-parameter model with 104B activated parameters, native multimodal input, Kimi Delta Attention (KDA), Attention Residuals, and Stable LatentMoE. Its original routed experts were trained with MXFP4 weights and MXFP8 activations. [Original model card][moonshot-card]

DeepSeek reports 552B backbone parameters plus 196B Engram conditional-memory parameters. Its causal encoder-decoder design activates 8B parameters during prefill and 16B during decode. CSA2 shares/compresses attention state; Single-Pass mHC handles residual mixing. These accounting categories differ from Kimi's total/activated counts, so their ratio is not an inference-speed prediction. [DeepSeek card][deepseek-card]

| Implementation property | Kimi K3 configuration | DeepSeek V4.1 Flash inference configuration |
| --- | --- | --- |
| Transformer depth | 93; first layer dense, remaining 92 MoE | 40 |
| Hidden width | 7,168 | 5,120 |
| Routed expert input/output width | 3,584 latent width | 5,120 |
| Expert intermediate width | 3,072 | 2,304 |
| Routed experts per layer / selected per token | 896 / 16 | 384 / 6 |
| Shared experts | 2 | 1 |
| Router score function / output multiplier | Sigmoid / 1.0 | Square root of softplus / 1.5 |
| Attention | 69 KDA, 24 full-attention layers; gated MLA | CSA2 and sliding-window attention |
| Expert nonlinearity | SiTU, beta 4; up-branch beta 25 | SwiGLU, clamp limit 10 |
| Vocabulary entries | 163,840 | 129,280 |
| Draft prediction layers | `num_nextn_predict_layers=0` | Three; separate 128-expert, top-3 draft MoEs |

Sources: [Kimi configuration][kimi-config], [DeepSeek configuration][deepseek-config], [DeepSeek architecture description][deepseek-card]. Kimi's configured attention layer lists are **one-based**; checkpoint layer tensor names are zero-based. Preserve that distinction when constructing an adapter.

Our engineering interpretation is that the useful common interface is a weighted sparse expert dispatch operation. The surrounding block, hidden-state representation, attention caches, residual mixing, preprocessing, and speculative decoder are model-specific. A new model directory and different expert count cannot make the present engine compatible.

## Router equivalence ends at the dispatch interface

The inspected Kimi gate projects the **full-width** state in FP32, applies sigmoid, and adds a learned correction bias for top-16 selection. Mixture weights come from the original unbiased scores, normalized over selected experts, then multiplied by 1.0. Configured group count is one, so the implementation's grouped selection branch is inactive. [Kimi gate and configuration][kimi-code]

Conceptually, for token state `h`:

```text
scores = sigmoid(W_router h)
ids = top16(scores + correction_bias)
weights = scores[ids] / (sum(scores[ids]) + epsilon)
```

DeepSeek also separates selection bias from mixture weights, but uses square-root-softplus scores and a 1.5 output scale. Its reference gate can select a separate bias for image-span tokens. Its expert applies asymmetric clipping before SwiGLU. [DeepSeek Gate and Expert][deepseek-code]

For this repository, reusing top-k plumbing is reasonable; reusing DeepSeek's score transformation, activation, scale, or numerical tolerances is not. Near-tied experts require tests of both selected IDs and weighted output: a tiny score error may change dispatch discontinuously.

### LatentMoE changes what expert saliency means

Kimi routes before the 7,168-to-3,584 projection. Experts execute in latent space; their weighted sum passes through RMS normalization and the projection back to model width. The shared branch receives the original full-width input and is added afterward. SiTU smooth-saturates both gate and up branches. [KimiSparseMoeBlock and SituAndMul][kimi-code]

This ordering creates an important research constraint. The inherited saliency heuristic measures expert contribution in its existing block geometry. A large latent-space expert norm need not imply an equally large final residual-stream contribution after normalization and projection. An initial Kimi trace should retain selected IDs, unbiased weights, latent outputs, and post-projection effects separately. Evaluate leave-one-expert-out changes to the complete block on sampled tokens before assuming the old score predicts damage.

Neither model's expert IDs represent shared semantic labels. DeepSeek's hot sets, topic labels, frequency rankings, and pruning masks cannot initialize a Kimi keep-set. Similar top-k fractions do not establish similar cache hit rates or tolerance of missing experts.

## NVFP4 is a storage and execution contract

NVIDIA's artifact converts original MXFP4 routed experts to NVFP4 with `input_scale=1.0`, and quantizes supported attention projections to FP8 in 128-by-128 blocks without calibration data. Remaining tensors retain their prior precision. This is a mixed checkpoint, not an entirely four-bit network. [NVIDIA card][nvidia-card]

The quantization manifest declares `MIXED_PRECISION`, NVFP4 expert groups of 16, and `FP8_PB_WO` attention targets. Exclusions include routers, shared experts, latent projections/norms, the first dense MLP, the LM head, vision, and multimodal projection. Its alias entries cover multiple runtime naming conventions; counting manifest targets does not count distinct tensors. [Quantization manifest][quant-config]

The existing [DeepSeek dequantizer](../../tools/v41_ref.py) expects packed E2M1 values with UE8M0 scales per 32 weights; its FP8 dense path uses 32-by-32 blocks. Consequently:

1. Build a tensor inventory from safetensors headers: names, shapes, dtypes, packing order, block scales, auxiliary scales, and offsets.
2. Implement a Kimi-specific reader and compare decoded sampled tensors with the pinned supported runtime. Do not reinterpret NVFP4 scale bytes using the existing UE8M0 decoder.
3. Verify SiTU and activation scaling before fusing the expert kernel. Matching packed weight values alone is insufficient.
4. Treat `FP8_PB_WO` separately from expert quantization; reject unknown schemes instead of silently using a DeepSeek fallback.

These are proposed adapter requirements, not implemented capabilities. NVIDIA's compatibility patch also addresses fused-projection alignment and FP8 dispatch, demonstrating that shape and dispatch details matter beyond the nominal bit width. [Runtime patch][patch]

## Resource implications for one Spark

The following arithmetic uses configuration dimensions, three expert matrices, four-bit values, and an illustrative one-byte scale per 16 weights. It excludes auxiliary tensor scales, alignment, allocator overhead, dense/shared/vision weights, caches, graph buffers, and activations. Actual checkpoint headers must replace these estimates before allocation decisions.

```text
routed expert instances = 92 * 896 = 82,432
weights per expert     = 3 * 3,584 * 3,072 = 33,030,144
packed value bytes     = 33,030,144 / 2 = 16,515,072
illustrative scales    = 33,030,144 / 16 = 2,064,384 bytes
estimated expert bytes = 18,579,456 bytes (~17.72 MiB)
routed-expert payload  = 82,432 * 18,579,456 = ~1.531 TB decimal
```

With a **hypothetical** 90 GB decimal arena, only about 4,844 experts (~5.88% of all instances) fit under those assumptions. This is not a proven available arena: Kimi's non-routed tensors and state could leave much less memory or already exceed the device budget. Reusing an inherited 40% resident target would demand roughly 612.6 GB for routed experts alone.

Top-16 selection across 92 MoE layers means 1,472 expert invocations per decode token. A pessimistic all-miss, no-reuse estimate is about **27.35 GB of expert reads per token**. At an illustrative effective storage bandwidth of 5 GB/s, that alone takes 5.47 seconds/token before compute and other work. This is a bandwidth thought experiment, not a throughput forecast: measured temporal locality, overlap, cache eviction, batching, and speculation can change the result.

For comparison, the inherited DeepSeek geometry has 15,360 routed expert instances and 240 expert invocations for a full 40-layer token pass. Kimi's sparsity fraction is about 1.79% versus 1.56%, yet it has over five times as many expert instances. The percentage alone hides the storage and active-work expansion.

NVIDIA reports validation on **eight B300 GPUs**, approximately **1.6 TB** download size, and a validated vLLM context setting of **196,608**, distinct from the architectural 1,048,576 limit. Its card identifies special runtime images/patch requirements. These results provide a reference deployment, not single-Spark support. [NVIDIA deployment notes][nvidia-card]

## Quality comparison and limits of published benchmarks

DeepSeek's published comparison reports K3 / V4.1 Flash scores of 92.9 / 90.9 on GPQA Diamond and 88.3 / 90.6 on Terminal-Bench 2.1. These establish no universal winner and are publisher-reported comparisons with differing model/harness histories. They do not measure this repository's streaming or pruned execution. [DeepSeek evaluation table][deepseek-card]

A fair local comparison requires the same task revision, harness, tools, token budget, sampling, and scored outputs, while reporting each model's own reasoning configuration. Quantization error, residency policy, and pruning error must be isolated in separate experiments. Resident caching with exact misses and unchanged arithmetic aims to preserve the checkpoint; dropping experts or substituting resident experts changes the model's computation.

Moonshot documents always-enabled thinking with `low`, `high`, or `max` effort and preservation of complete assistant reasoning/tool-call history between turns. [Kimi usage instructions][moonshot-card] The inherited server's thinking-off default and DeepSeek encoding therefore require a separate Kimi protocol adapter, not just a renamed model response field.

## Recommended reuse boundary

| Existing investment | Kimi treatment |
| --- | --- |
| NVMe I/O, arena/LRU concept, hit/miss counters | Reuse design after variable tensor sizes and scale metadata are supported |
| Trace capture and workload stratification | Reuse methodology; collect fresh Kimi traces and revise saliency definitions |
| Prune/substitute/drop experiments | Keep disabled until a correct exact-routing baseline exists |
| DeepSeek expert math and dense quantization | Replace with validated Kimi SiTU/NVFP4/FP8 paths |
| mHC, CED/CSA2 state, Engram | Keep in DeepSeek implementation; introduce KDA/MLA/AttnRes separately |
| DSpark draft weights and verification state | Do not reuse; independently qualify any Kimi draft model and rollback protocol |
| Chat/tool parser and tokenizer | Implement Kimi-specific protocol fixtures and round-trip checks |

The decisive next milestone is a reproducible tensor/memory inventory and numerically validated Kimi block, followed by an exact-routing end-to-end baseline. A single-device pruning target should follow that evidence, not precede it. See the [implementation plan](implementation-plan.md).

[nvidia-card]: https://huggingface.co/nvidia/Kimi-K3-NVFP4/blob/b2428a0b83a8b712ff2e1a8448a103d4175341f1/README.md
[kimi-config]: https://huggingface.co/nvidia/Kimi-K3-NVFP4/blob/b2428a0b83a8b712ff2e1a8448a103d4175341f1/config.json
[quant-config]: https://huggingface.co/nvidia/Kimi-K3-NVFP4/blob/b2428a0b83a8b712ff2e1a8448a103d4175341f1/hf_quant_config.json
[kimi-code]: https://huggingface.co/nvidia/Kimi-K3-NVFP4/blob/b2428a0b83a8b712ff2e1a8448a103d4175341f1/modeling_kimi_linear.py
[patch]: https://huggingface.co/nvidia/Kimi-K3-NVFP4/blob/b2428a0b83a8b712ff2e1a8448a103d4175341f1/runtime_patches/sitecustomize.py
[moonshot-card]: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/README.md
[deepseek-card]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/README.md
[deepseek-config]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/config.json
[deepseek-code]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py

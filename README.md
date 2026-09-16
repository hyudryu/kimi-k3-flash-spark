# Kimi K3 NVFP4 on NVIDIA DGX Spark - work in progress

> **DO NOT USE. This is one of my 50 WIPs.** It is unfinished research, is not
> production-ready, and does not yet implement or validate Kimi K3 inference.

This repository is investigating a DGX Spark implementation for
[NVIDIA Kimi-K3-NVFP4](https://huggingface.co/nvidia/Kimi-K3-NVFP4), starting from
an inherited DeepSeek-V4.1-Flash engine that combines measured expert selection,
a resident expert cache, and NVMe streaming.

**The project target is Kimi K3 NVFP4; the executable code is still DeepSeek-specific.**
Changing `MODEL_DIR` to a Kimi checkpoint is not a supported conversion. The model
loader, attention, routing, quantization kernels, tokenizer, and tool protocol all
need compatibility work. No Kimi throughput, quality, memory-fit, or single-Spark
serving result has been measured in this repository.

## Research and implementation plan

- [Research index and scope](docs/kimi/README.md)
- [Repository methodology and expert selection](docs/kimi/repository-methodology.md)
- [Kimi K3 NVFP4 versus DeepSeek-V4.1-Flash](docs/kimi/model-comparison.md)
- [Implementation plan and acceptance gates](docs/kimi/implementation-plan.md)

The research distinguishes model routing (which experts a token needs), cache
placement (where those experts live), and pruning (which experts remain eligible).
Streaming the original experts and pruning/repacking them have different quality
implications; they must not share an unqualified "full quality" claim.

## Current code and inherited evidence

| Area | Current state |
| --- | --- |
| Engine | DeepSeek-V4.1-Flash, including its attention, Engram, and DSpark paths |
| Expert storage | Native DeepSeek FP4 streaming and experimental CB3/pruned resident modes |
| HTTP API | OpenAI-compatible transport with DeepSeek-specific prompt/tool handling |
| Launch scripts and container | Inherited DeepSeek configuration; not Kimi installation instructions |
| Kimi K3 NVFP4 | Research and implementation plan only |

Historical DeepSeek measurements remain in [RESULTS.md](RESULTS.md),
[LIMITATIONS.md](LIMITATIONS.md), and [NOTES.md](NOTES.md). They are not Kimi
benchmarks, and configurations and limitations differ across dated runs.
The earlier README is preserved in
[the inherited revision](https://github.com/hyudryu/kimi-k3-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/README.md).

Existing [architecture](docs/architecture.md), [installation](docs/install.md),
[expert keep-set](docs/keep-sets.md), [tuning](docs/tune.md), and
[benchmarking](docs/benchmarking.md) documentation describes that DeepSeek work.
Read the new research before treating any of it as applicable to Kimi.

## Attribution and licensing

See [CREDITS.md](CREDITS.md) for inherited work and [LICENSE](LICENSE) for this
repository's code license. Model weights have their own terms; consult the
[NVIDIA model card](https://huggingface.co/nvidia/Kimi-K3-NVFP4) and its linked
upstream terms before obtaining or redistributing them.

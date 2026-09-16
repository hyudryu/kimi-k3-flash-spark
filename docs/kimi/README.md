# Kimi K3 NVFP4 research

> **Do not use: one of my 50 WIPs.** These documents plan a port; they do not
> establish Kimi runtime support.

Research date: 2026-09-15. Repository baseline:
`45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f` (inherited release 0.5.0).

1. [Repository methodology](repository-methodology.md): the actual execution path,
   expert selection, storage policies, calibration, and quality tradeoffs.
2. [Model comparison](model-comparison.md): primary-source Kimi/DeepSeek evidence,
   architectural differences, quantization compatibility, and open questions.
3. [Implementation plan](implementation-plan.md): proposed work packages, dependencies,
   decision gates, and the measurements required before claiming success.

The scope is a feasibility study for the NVIDIA checkpoint on DGX Spark, beginning
with text generation. Native multimodal capability is not evidence that this
repository's text-only engine supports image or video inputs. Existing DeepSeek
traces, keep sets, and performance figures are not transferable Kimi evidence.

Read source facts, historical measurements, engineering estimates, and proposed
changes as separate categories. Remote `main` links can change; the comparison
records the revisions inspected wherever available.

# Repository methodology and expert selection

Source audit: 2026-09-15. This describes the checked-in implementation, not a tested Kimi K3 port. **Do not use this WIP as a supported Kimi runtime.** Renaming the project does not make its checkpoint loader, kernels, router, or chat protocol compatible with Kimi.

## What is implemented

The repository is a specialized, single-sequence DeepSeek-V4.1-Flash inference stack for a Linux NVIDIA DGX Spark/GB10. Its central constraint is storing a model whose routed expert weights exceed unified memory. It offers two substantially different strategies: retain the original expert choices and stream missing weights, or restrict the expert population and accept a changed model. Additional quantization and speculative decoding are independent dimensions.

| Component | Implementation and responsibility | Portability assessment |
|---|---|---|
| HTTP/API layer | [`server/app.py`](../../server/app.py) and [`Engine`](../../server/engine_api.py): request serialization, streaming token bursts, reasoning/tool parsing, cancellation | Transport concepts are reusable; current encoding, DSML grammar, EOS and reasoning conventions are model-specific |
| Runtime coordinator | [`V41Engine`](../../engine/v41_engine.py#L406): initialization, budgets, keep-set construction, generation, sampling, speculation and statistics | Useful orchestration pattern; not a generic model loader |
| Transformer execution | [`Model`](../../engine/model.py): compressed/sliding-window attention, shared KV producers, hyper-connections, Engram and DSpark | Requires an architecture-specific implementation or adapter |
| Expert storage | [`ExpertStore`](../../engine/experts.py#L113): file ranges, staging buffers, resident slots, LRU and transient ring | Residency policy is reusable; tensor names, sizes and expert counts are specialized |
| MoE compute | [`fp4_moe.py`](../../tools/fp4_moe.py), [`cb3_moe.py`](../../tools/cb3_moe.py), [`moe_fallback.py`](../../engine/moe_fallback.py) | FP4 and CB3 layouts are explicit contracts, not arbitrary four-bit checkpoint support |
| Fast decode | [`FastDecoder`](../../engine/fastdecode.py): static buffers, CUDA graphs and a resident slot lookup table | Depends on fixed dimensions, routing semantics and verification shape |
| Offline profiling | [`expert_trace.py`](../../tools/expert_trace.py), [`expert_stats.py`](../../tools/expert_stats.py) | Corpus/profile workflow is reusable after model-specific tracing is replaced |
| Budget/profile tooling | [`budget.py`](../../tools/budget.py), [`tune.py`](../../tools/tune.py) | Ranking concepts transfer; byte counts and calibrated memory formulas do not |

The loader reads `inference/config.json`, DeepSeek-style names such as `layers.L.ffn.experts.E.w1.weight`, and separately loads three DSpark expert blocks. The server imports the checkpoint's `encoding/encoding.py`. These requirements alone prevent treating a different Hugging Face model directory as a drop-in replacement. See [`V41Engine.__init__`](../../engine/v41_engine.py#L410), [`Weights`](../../engine/model.py#L78), and [`load_encoding_module`](../../server/app.py#L110).

## Three different meanings of expert selection

1. **Neural routing:** which experts the trained gate selects for a particular token. It changes at every layer and token.
2. **Storage residency:** which selected experts happen to have loaded weights. In unpruned streaming mode a cache miss loads the requested expert; it does not authorize substituting a different one.
3. **Offline pruning:** which experts are allowed to participate, based on a profiled corpus and a memory budget. This changes the model's computation even when surviving weights retain their original precision.

The heading “A keep-set is a cache policy” in [`docs/keep-sets.md`](../keep-sets.md) conflates the last two. In this implementation a pruning mask changes the routing result or removes contributions. It is lossy model modification, not merely caching. Similarly, CB3 conversion changes weight values independently of pruning. Quality must be evaluated separately for each change.

## Exact backbone routing behavior

[`Model.moe`](../../engine/model.py#L482) computes the following for a token representation `x` and expert `e`:

```text
z_e = (x W_gate)_e                         # float32 gate computation
s_e = sqrt(softplus(z_e))
q_e = s_e + bias_e                         # selection score only
I   = top_k(q)
g_e = route_scale * s_e / (sum_{j in I} s_j + 1e-20), for e in I
y   = shared_expert(x) + sum_{e in I} g_e * expert_e(x)
```

The bias affects selection, **not** the mixture weights. This is not sigmoid or softmax routing. The checked-in [`Args`](../../tools/v41_ref.py#L233) defaults are 40 backbone layers, 384 routed experts per layer, top-6, hidden width 5,120, expert intermediate width 2,304 and `route_scale=1.5`. Configuration can override some `Args` fields, but many downstream arrays and loops still hard-code those dimensions. `DSV41_TOPK` is a separate experimental reduction of backbone top-k; DSpark remains top-3 of 128.

The fast path repeats the routing logic in [`FastDecoder._layer_a`](../../engine/fastdecode.py#L322). Both implementations must stay aligned. [`test_route_modes.py`](../../tools/test_route_modes.py) checks a NumPy reference and source-pattern assertions; that is useful consistency coverage, but not a GPU execution equivalence proof.

### Substitute mode

With a pruning keep-set `K`, the default `substitute` mode replaces `q_e` with negative infinity for every `e` outside `K` **before** top-k. It selects top-k from the survivors and renormalizes their original positive `s_e` values to `route_scale`. Some selected experts may differ from the full model's choices. Keeping six experts active does not mean preserving the full model's six experts.

### Drop mode

The experimental `drop` mode first obtains the full model's top-k, then zeros the weights of picks outside `K`. It renormalizes surviving weights to `route_scale`; if all picks were removed, the routed contribution is zero and only the shared expert remains. This is a variable effective top-k, not uniform attenuation of the routed branch.

Zero-weight picks are remapped to distinct resident fallback expert IDs for safe slot lookup. Otherwise the storage layer would fetch an excluded expert or use an invalid slot despite its zero weight. [`build_prune_fallback`](../../engine/v41_engine.py#L378) and the fast path enforce bounds needed by the small-routing kernel. These fallback IDs are storage placeholders, not the original semantic routing choices. The kernel can still execute work for zero-weight placeholders, so nominally fewer nonzero contributions do not directly imply proportional speedup.

`env.example` records a negative historical generation experiment for `drop`; it remains commented out. Its conceptual appeal does not establish better quality.

## How offline keep-sets are learned

### Trace collection

[`expert_trace.py`](../../tools/expert_trace.py) processes a tagged, teacher-forced corpus through the reference-style implementation one layer at a time. Per-layer `.npz` artifacts contain top-k IDs, routing weights, category labels and weighted expert contribution norms. [`v41_ref.block_forward`](../../tools/v41_ref.py#L735) measures the norm of the contribution actually computed, including the implementation's rounding.

For a topic `t`, layer `l` and expert `e`, statistics supply two choices:

```text
C[t,l,e] = sum over tokens i in topic t of 1[e was selected for i]
S[t,l,e] = sum over selected occurrences i of ||g[i,e] * E_e(x_i)||_2
```

`counts` ranks frequency; `saliency` ranks accumulated weighted output magnitude. This implementation sums contributions; it does not divide each expert's score by the number of times that expert fired. Thus it combines frequency and magnitude. Comments attribute this to REAP-inspired pruning, but this audit establishes the implemented statistic, not equivalence to a paper's algorithm or its Kimi benchmark results. See [`saliency_hist`](../../tools/expert_stats.py#L65).

Current traces store `contrib_norms` in float32. Older `out_norms` traces are converted by multiplying routing weights; non-finite legacy contributions are clamped and counted. A requested saliency profile must exist for all 40 layers or initialization raises; the engine does not silently use frequency instead. Old comments still saying only `out_norms` should not be read as the current artifact schema.

### Combining topics

Each selected topic is normalized independently within each layer:

```text
p[t,l,e] = H[t,l,e] / sum_j H[t,l,j]         # H is C or S; zero total stays zero
```

This gives each represented topic equal aggregate influence under `sum`, regardless of corpus token count. Topic names therefore matter: splitting one domain into many categories can increase its effective voting weight.

[`V41Engine`](../../engine/v41_engine.py#L610) implements three ranking rules:

| Rule | Computation | Meaning and limitation |
|---|---|---|
| `sum` | `score[l,e] = sum_t p[t,l,e]` | Maximizes summed selected-topic mass for a fixed per-layer count; a broadly distributed topic can lose to concentrated topics |
| `max` | `score[l,e] = max_t p[t,l,e]` | Retains experts important to at least one topic; does not optimize minimum topic coverage |
| `maxmin` | Repeatedly pick the currently least-covered nonzero topic, then admit its highest-ranked unselected expert; credit that expert to every topic | Greedy balancing heuristic, not a proof of globally optimal max-min coverage or generation quality |

[`_maxmin_counts`](../../engine/v41_engine.py#L284) encodes admission order as artificial scores above all excluded experts. These scores are not probabilities. All-zero topics do not vote. Equal-score and equal-coverage cases inherit sorting/topic-order behavior, so stable profile manifests should record topic ordering as well as the rule.

### Turning a ranking into a memory allocation

[`build_keep_masks`](../../engine/v41_engine.py#L339) computes `N = max(6, ceil(384 * keep_fraction))`.

* `uniform` keeps the top `N` experts in each layer.
* `global` reserves 24 experts per layer, normalizes each layer's supplied score vector, then allocates remaining slots by a cross-layer ranking under nominal budget `N * number_of_layers`.
* `maxmin` with `global` is explicitly rejected: admission-order scores cannot be compared as cross-layer mass.

The global helper assumes usable nonzero histograms and a budget large enough for its per-layer floor; it does not robustly validate all adversarial inputs. At very small keep fractions, the 24-per-layer floor can already exceed the nominal budget. Porting should add shape, finiteness, zero-mass and floor-versus-budget validation rather than generalizing these assumptions unchanged.

Selected experts are warm-loaded into the LRU. “Pruned all-resident” is an intended configuration, not an unconditional guarantee: the engine warns when the selected count exceeds LRU capacity and the tail must stream. The device slot lookup optimization is enabled only when the residency condition is met. A small transient ring is safe only when the active set is fully resident or the maximum miss set fits the ring.

[`budget.py`](../../tools/budget.py) separately implements ranking for the tuning interface. [`test_budget_rank.py`](../../tools/test_budget_rank.py) compares its selected sets with the engine's AST-extracted implementation. Both must change together in a port.

## Residency and I/O methodology

The packed DeepSeek FP4 expert contains three matrices and their scales: two logical `2304 x 5120` up/gate matrices and one `5120 x 2304` down matrix. Two E2M1 values occupy one byte; each group of 32 weights has a UE8M0 scale. [`ExpertArena`](../../tools/fp4_moe.py#L81) uses **18,800,640 bytes per expert**, or about 288.8 decimal GB for 40 x 384 experts. This byte format is the repository's DeepSeek format; a Kimi NVFP4 label alone does not establish scale/layout compatibility.

[`ExpertStore.resolve`](../../engine/experts.py#L368) deduplicates requested expert IDs and reserves all already-resident slots before loading misses. It protects slots used by the current call to prevent eviction collisions. Decode misses enter the LRU; prefill misses enter a separate transient ring so broad prompt routing does not evict the decode working set. A transient hit used during decode can be promoted to the LRU by swapping slot ownership without rereading weights.

[`ShardFile.expert_runs`](../../engine/experts.py#L70) merges adjacent tensor spans into maximal contiguous ranges. The historical checkpoint's six tensors typically form two runs; this is a property of its serialization, not a universal MoE invariant. Linux `O_DIRECT`/`preadv`, aligned pinned staging buffers, an expert-level thread pool and a second read-chunk pool move those ranges into arena slots. The separate pools avoid nested-task deadlock. This is not a native Windows I/O path.

Unpruned warm-start ranking uses [`rank_from_trace`](../../engine/experts.py#L518), primarily frequency normalized by layer, with missing layers interleaved into fallback ranking. A poor warm set affects performance; it does not itself alter the original neural routing. Exact end-to-end numerical equivalence still depends on kernels, optional quantization and attention execution.

`CB3` is a separate on-load conversion into three-bit per-row codebook indices plus original group scales. [`CB3_BYTES_PER_SLOT`](../../tools/cb3_moe.py#L27) evaluates to **14,454,784 bytes**, about 76.9% of the FP4 storage. This permits more resident experts but introduces a separate quantization approximation and packing cost. The file's old “3.07 bpw” introductory label excludes/conflicts with full overhead arithmetic: with 35,389,440 logical expert weights, total storage is approximately **3.27 bits per weight**. Use the executable byte constant, not that label, for budgets.

## Defaults: bare engine versus example deployment

These are configuration observations, not recommended Kimi settings. See [`V41Engine.__init__`](../../engine/v41_engine.py#L410) and [`env.example`](../../env.example).

| Axis | Bare engine / unset environment | Values explicitly shipped in `env.example` |
|---|---|---|
| Pruning | Off (`prune_keep=None`) | `PRUNE_KEEP=0.39`: 150 experts/layer, 6,000 total |
| Expert representation | FP4 | CB3 |
| Arena budget | Automatic sizing | 87 decimal GB |
| Selection | Uniform | Empty `PRUNE_SELECT`, interpreted by launchers as uniform |
| Ranking / measurement | `sum` / `counts` | `maxmin` / `saliency` |
| Non-kept picks | `substitute` | Still `substitute`; `drop` is commented out |
| Transient slots | 400 | 8 |
| Free-memory floor input | 20 GB | 6 GB, with additional prefill/watchdog checks |
| Dense projection conversion | No dense FP4 groups | `attn,wo_a` |
| LM head | BF16 unless configured otherwise | FP8 |
| Context allocation | 32,768 | 32,768 |
| Speculative decoding | Enabled | `SPEC=1` |

Consequently, “checkpoint precision is preserved” applies only to specific FP4 storage paths, not the complete example deployment. Copying the example activates pruning, CB3 expert conversion, dense FP4 conversion and FP8 head conversion together.

## Attention, speculation and memory constraints

The model implementation includes compressed global KV, a 128-token sliding window, cross-layer KV producers, hyper-connection mixing and Engram token-history lookups. Its bounded sliding-window replay skips full-prompt execution for some decoder layers. These optimizations depend on the current model architecture; similar MoE feed-forward blocks do not justify transferring them to another attention or residual design.

DSpark uses three resident top-3/128-expert blocks, trained to propose five draft tokens; the target verifies a six-token block including the already selected token. Throughput depends on both bytes touched by the union of experts in that block and draft acceptance. The presence of speculative verification is not a standalone proof that the implemented fast path is numerically lossless. [`test_spec_lossless.py`](../../engine/test_spec_lossless.py) is the relevant greedy-output comparison; teacher-forced loss does not execute the full generation loop.

[`V41Engine`](../../engine/v41_engine.py#L494) consults both CUDA free memory and Linux `MemAvailable`, reserves prefill/packing space and starts a memory watchdog. The prefill reserve is an empirical DeepSeek/GB10 formula using chunk length and context length. [`budget.py`](../../tools/budget.py#L204) labels 131,072 as its validated maximum sequence calibration; later historical notes describe additional experiments. Neither label is a universal KV-memory formula or Kimi capacity guarantee.

## What the measurements do and do not establish

[`RESULTS.md`](../../RESULTS.md), [`NOTES.md`](../../NOTES.md), [`LIMITATIONS.md`](../../LIMITATIONS.md) and [`results/keepsets`](../../results/keepsets/README.md) are a historical experimental ledger. Earlier entries can be superseded by later ones: for example, “CB3 simulation only” and “tool grammar unverified” do not describe every later implementation/experiment. Conversely, a later successful short-prompt test does not establish long-context safety or broad task quality.

Important limits of the methodology:

* **Coverage is a proxy.** Routing-slot coverage and contribution-magnitude coverage are different metrics; neither measures correctness, calibration, tool validity or generation stability.
* **Teacher forcing hides feedback.** A keep-set measured on unmodified activations need not induce the same routing after pruning changes earlier layers. Repeated free-generation profiling and held-out evaluation are needed.
* **Norm saliency ignores direction.** Large individual contributions can cancel or be redundant; low-magnitude experts can be decisive for rare tokens. It is not a causal importance score.
* **Topics are corpus-dependent.** Missing tool boundaries, reasoning exits, markup, languages or domains are not repaired by choosing a clever ranking rule. Normalization cannot recover absent evidence.
* **Trace provenance matters.** Model revision, router settings, precision, tokenizer, categories and corpus hashes should accompany each artifact. Older logs explicitly note traces predating numerical fixes.
* **Single-call safety is not serving scale.** The API serializes generation, and the engine is batch size one. Cache and memory results do not imply multi-request or batched capacity.
* **Tests have different scopes.** AST/source checks verify wiring; kernel tests verify numerical pieces; speculative A/B checks generation consistency; task gates test a finite prompt set. None substitutes for the others.
* **Historical benchmark speeds are not Kimi speeds.** This documentation task has not loaded either checkpoint or reproduced GPU inference. No Kimi throughput, memory fit or quality result is established here.

The transferable methodology is to profile actual routing, separate storage policy from model changes, budget all memory consumers, optimize measured I/O/compute bottlenecks, and gate each approximation with generation tests. The current constants, checkpoint assumptions and keep-sets must be replaced or explicitly validated before applying that methodology to Kimi K3 NVFP4.

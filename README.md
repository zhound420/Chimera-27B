# Chimera-27B v0.1

Runtime patches, reproducible evaluation, and reconstruction tools for the experimental model.

**[Download model weights on Hugging Face](https://huggingface.co/zhound/Chimera-27B-v0.1-GGUF/tree/v0.1).** This GitHub repository contains code and results; large model files are hosted on Hugging Face.

**An experimental Qwen 27B backbone with one calibrated DeepSeek shared expert added at layer 0. A custom runtime is required.**

This release demonstrates a functioning cross-family weight graft. It does **not** establish that the graft improves quality, and it is **not** a conversion of the dense backbone to sparse MoE. The full original Qwen computation remains active. This is an independent release by [zhound](https://huggingface.co/zhound), with substantial AI assistance in implementation, calibration, testing, and documentation.

**Do not load these files in an unpatched engine and assume the hybrid is active. Stock engines may ignore the extra donor tensors.** Use the pinned runtime patch and build instructions below. The automated Hub library examples, if displayed, are not instructions for this custom architecture. No hosted inference endpoint is provided.

## Downloads

| File | Size (decimal GB) | Contents |
|---|---:|---|
| `Chimera-27B-v0.1-Q4_K_M.gguf` | 16.848 | Q4_K_M backbone, Q8_0 donor |
| `Chimera-27B-v0.1-Q8_0.gguf` | 29.085 | Q8_0 backbone and donor |
| `mmproj-Qwen3.8-27B-BF16.gguf` | 0.931 | Qwen vision encoder/projector |
| `reconstruction/shared-aligned-fp32.npz` | 0.142 | Calibrated donor before Q8_0 quantization |

The files retain their evaluated bytes and historical internal model names. Renaming the files to Chimera did not modify the GGUF contents. Both editions require the separate projector for images. File size is not total inference memory: KV cache, recurrent state, MTP, vision, and runtime buffers also consume memory.

## What changed

The base is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. The donor is the **layer-0 shared expert** from [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.

Three added projections contain **35,389,440 donor parameters**: hidden width 5,120 and intermediate width 2,304. Linear input/output maps were fitted with ridge regression and folded into the donor projections. The patched layer computes:

```text
FFN_output = original_Qwen_FFN(x) + 0.1 * aligned_DeepSeek_shared_expert(x)
```

The donor retains DeepSeek's clamped SwiGLU semantics, with clamp 10. The original Qwen tensor shapes, attention, tokenizer, vision encoder, and MTP head are retained. There is no learned router and no gradient training of the combined model. The donor was selected as an interface probe, not as a proven best coding expert.

Calibration used paired text-span activations from C++ source excerpts: 2,048 fitting pairs and 1,386 validation pairs separated by source file. A quantized DeepSeek Coder backbone supplied the activation proxy; the donated weights themselves came from the official DeepSeek model. Ridge strengths were selected on the alignment validation set, so those alignment errors are not independent final test scores. The output map used residual representations as a proxy; good alignment error alone does not demonstrate useful capability transfer. See [provenance.json](provenance.json) and [alignment results](results/alignment-result.json).

## What was measured

These are **small local screens, not established coding benchmark scores**. There were eight development coding tasks and three subsequently introduced coding tasks, one seed (42), temperature 1.0, top-p 0.95, top-k 20, and medium reasoning for Qwen/Chimera. Initial output budgets were 4,096 tokens, including reasoning. Swift used high reasoning. The same published tasks must not be treated as unseen evaluation data in future experiments.

| Model | Development | Three subsequent tasks | Supplementary retests at 16,384 output tokens |
|---|---:|---:|---|
| Qwen Q8 baseline | 8/8 | 2/3 | Previously truncated parser passed |
| Chimera Q8 | 8/8 | 3/3 | None needed |
| Chimera Q4, tuned | 7/8 | 2/3 | Both truncated cases passed: JSON patch, parser |
| Swift reference | Not part of this matched development comparison | 2/3 | None; failed transaction case had a code error |

The supplementary retests are **not additional independent cases or an 11/11 pass@1 result for Q4**. They resolve output-budget failures. Plain Qwen also completed its truncated case when given more output tokens. These results do not establish a quality advantage for Chimera.

Q8 and tuned Q4 passed one tool-retry fixture and one vision fixture. Thinking was enabled. A 131,072-token context allocation was used, but end-to-end 128K retrieval/reasoning quality was **not tested**. Repository editing, long agent trajectories, broader vision, multiple seeds, and donor-on/off task ablations remain outstanding.

The zero-donor GPU control matched all three short continuation probes, with zero difference in the recorded chosen-token log probabilities. This checks the separate-branch implementation on those probes; it does not prove universal equivalence or donor usefulness.

### Throughput

The best selected Q4 configuration reached **69.009 tokens/second**, the median of three 768-token coding decode probes. It was selected from 16 successfully loaded configurations; a seventeenth row-split configuration failed to load. Probes can end at the output limit and are not quality scores. This is decode speed, not prefill speed or end-to-end agent task latency. It is not sustained long-context throughput, and the 80 TPS target was not reached.

The selected configuration uses five MTP drafts and an optional CUDA dispatch shim with cutoff 2. Q4 without MTP measured about 40 TPS. Q8 observed about 26 TPS on its quality screen. Improvements are attributable to quantization and runtime tuning; no speedup attributable to the donor graft has been established. Speed depends on prompt, output, sampling, hardware, and concurrent activity.

Reference machine:

- NVIDIA RTX 3090, 24 GiB, plus Tesla PG500-216, 32 GiB; PCIe 3.0 x4 links, no NVLink.
- Four Xeon E5-4640 v1 processors, 32 physical cores / 64 threads; approximately 117 GiB RAM.
- CUDA 12.8, GCC 13, Linux, NVMe model storage.
- Q4 language inference on the RTX 3090; vision on the Tesla. Q8 used a 0.40/0.60 layer split with vision on GPU 0.
- One request slot, Q8_0 K/V cache, flash attention, batch 256 / microbatch 128, eight CPU threads.

The [results directory](results) contains raw saved responses, fixtures, timings, and primary/retest results. Absolute machine paths are redacted; historical run labels remain. The machine's private service configuration and credentials are not part of this release.

## Build the required runtime

The reference workflow is Linux with Python 3.12+, CMake, Ninja, a C++ compiler, `patch`, and a CUDA toolkit appropriate for your GPU. CUDA 12.8 is needed to reproduce the original sm_70 + sm_86 build; newer toolkits may drop older GPU architectures. No precompiled machine-specific binaries are shipped.

Clone the source and download Q4 plus the vision projector from the pinned Hugging Face release:

```bash
git clone https://github.com/zhound420/Chimera-27B.git
cd Chimera-27B
python3 download_weights.py --precision q4
python3 runtime/build.py --architectures '70;86' --jobs 4
```

`download_weights.py` requires the Hugging Face `hf` CLI and verifies each downloaded file against `weights-manifest.json`. Use `--precision q8` for Q8, `--precision both` for both editions, or add `--donor` for reconstruction. `--dry-run` asks the CLI to list the planned downloads without transferring the weights. `SOURCE_SHA256SUMS` covers the GitHub source snapshot; the separate weight manifest records the Hugging Face artifacts.

The build script downloads and checksums llama.cpp revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`, applies [runtime/chimera.patch](runtime/chimera.patch), then builds the server, quantizer, and optional Linux dispatch shim. `--cuda-root /path/to/cuda` selects a toolkit; `--cc gcc-13 --cxx g++-13` selects compilers. Use architectures appropriate for your GPUs. `--legacy-cpu` disables AVX2/FMA/F16C on hosts such as the original E5-v1 machine. `--cpu-only` builds a CPU backend, but the provided serving profiles target CUDA.

Release validation rebuilds the patched model/server code from a fresh pinned source tree while reusing the matching prebuilt GGML SDK on the original host. It does not claim a second full CUDA-kernel rebuild on an unrelated machine. See [release-validation.json](release-validation.json).

## Run

For the measured two-GPU Q4 configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 runtime/serve.py --precision q4 --mmvq-cutoff 2
```

The launcher defaults to localhost, port 8081, medium reasoning, 131,072 context, one slot, five MTP drafts, language on CUDA0, and vision on CUDA1. It does not stop other model servers or alter system services; sufficient free VRAM must already be available. Use `--host 0.0.0.0` only when you intend to expose the unauthenticated local API to your network.

The optional dispatch shim redirects a narrow set of quantized Ampere (compute capability 8.6) operations from MMVQ to MMQ. It is specific to the pinned runtime's Linux ABI. Omit `--mmvq-cutoff` for the ordinary kernel selection; other hardware may have a different optimum. All benchmark numbers for the selected Q4 configuration used cutoff 2.

For Q8 on the original two GPUs:

```bash
python3 runtime/serve.py --precision q8
```

For a single GPU with enough VRAM, Q4 can put vision on the same GPU using `--vision-device CUDA0`; memory use and performance for that configuration are not validated here. `--context`, `--threads`, `--draft-max`, and `--tensor-split` are adjustable. `--donor-scale 0` disables the donor contribution for ablations. Changing these settings requires new measurements.

API settings:

```text
Base URL: http://127.0.0.1:8081/v1
Model: chimera-27b-v0.1
API key: none (use a placeholder if your client insists)
Reasoning effort: medium
Suggested max output: 16384
Temperature: 1.0; top_p: 0.95; top_k: 20; min_p: 0.0
```

Prompt tokens plus the requested output budget must fit within the context window. A 128K context setting is not a 128K output allowance. Images use OpenAI-compatible `image_url` messages; tool calling uses the compatible `tools` schema. The server must load the projector for vision.

## Reproduce the checks

The coding harness uses Bubblewrap to execute generated Python with no network, limited memory/CPU, and read-only system files. Install `bubblewrap`; it must support unprivileged user namespaces on your host. There is no unsandboxed fallback.

With the patched server running:

```bash
python3 evaluation/baseline.py --model chimera-27b-v0.1 --label development-reproduction --effort medium --max-tokens 4096
python3 evaluation/heldout.py --model chimera-27b-v0.1 --label subsequent-reproduction --effort medium --max-tokens 4096 --coding-only --split heldout
python3 evaluation/throughput.py --model chimera-27b-v0.1 --output-dir ./throughput-reproduction --draft-max 5
```

Use fresh labels to preserve prior outputs. `--case-names json_patch` or `--case-names exact_arithmetic` and `--max-tokens 16384` reproduce the larger-budget retest procedure with the corresponding harness. Different GPU math or sampling paths can produce different answers despite the same seed.

To re-execute the published coding responses without running a model:

```bash
python3 evaluation/replay.py
```

This verifies the recorded executable outcomes, not general model quality. The published primary and supplementary results remain separate. Tool and vision response logs are included; replay.py replays coding checks only.

## Reconstruct the weight graft

Install NumPy and the `gguf-py` package from the pinned runtime source. Convert the pinned official Qwen checkpoint with that revision's `convert_hf_to_gguf.py`, then quantize the BF16 GGUF directly to Q4_K_M or Q8_0 with `llama-quantize`. Do not quantize the existing Q8 hybrid down to Q4 if you want to reproduce this release: Q4 was quantized directly from BF16 before appending the donor.

```bash
python3 download_weights.py --precision q4 --donor
python3 -m pip install numpy ./runtime/upstream/gguf-py
python3 reconstruction/append_donor.py --base /path/to/Qwen3.8-27B-Q4_K_M.gguf --output /path/to/rebuilt-Chimera-Q4_K_M.gguf
```

This reconstructs the graft using the supplied fitted donor, checks original tensor shapes, and checks the original layer-0 FFN bytes. The packaged script rebuilt a Q4 GGUF that compared byte-for-byte equal to the released Q4 file. It is not a complete automated rerun of the activation-capture/calibration pipeline; that used a separate instrumented DeepSeek runtime and quantized proxy. The calibration source excerpts are included in `reconstruction/alignment-corpus.json` and `.txt`, with their upstream MIT license retained. Exact source revisions, the alignment method, and limitations are recorded in the release.

## License and limits

Qwen-derived weights are under Apache-2.0; DeepSeek donor weights retain their MIT license. Copies and attribution are included in [LICENSE](LICENSE), [NOTICE](NOTICE), and [licenses](licenses). The runtime patch derives from MIT-licensed llama.cpp. New release scripts and documentation are Apache-2.0. Preserve these notices when redistributing.

Chimera is suitable for inspecting and reproducing this experiment. It should not be presented as a proven best-of-both model, an 80+ TPS result, a validated 128K agent, or a sparse MoE conversion. Multi-seed donor ablations and broader coding, tool, vision, and long-context evaluation are the next research steps.

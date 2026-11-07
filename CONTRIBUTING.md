# Contributing

This project studies whether a calibrated expert graft transfers useful behavior between model families. It is an independent project, not an upstream llama.cpp contribution.

For a proposed change, include the exact parent revisions, donor selection, runtime settings, and executable evaluation results. Compare against a donor-disabled control at the same quantization and sampling settings. Report unsuccessful cases and output truncation separately; larger-budget retries are not new pass@1 successes.

Keep calibration and model selection separate from final evaluation. The published v0.1 tasks are development fixtures and must not be described as unseen tests in future releases. Small local screens cannot establish broad coding, agentic, vision, or long-context superiority.

Model files belong on Hugging Face. Preserve the parent licenses and notices. Do not commit credentials, machine-specific deployment configuration, private prompts, or generated build directories. Disclose substantial AI assistance.

Before changing runtime math, verify the zero-donor control and retain the original backbone. Reproduce the published coding outcomes with `python3 evaluation/replay.py`; fresh model evaluation requires the patched server. A contribution should explain what improved and what remains untested.

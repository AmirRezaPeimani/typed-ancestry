# Third-Party Resources

No third-party dataset rows, model weights, tokenizer files, or checkpoints
are distributed in this repository.

## Datasets

| Resource | Pinned revision | License | Use |
|---|---|---|---|
| [nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3/tree/f6d145777bcbde96137596340fab89793acd1031) | `f6d145777bcbde96137596340fab89793acd1031` | CC BY 4.0 | Primary multi-view dataset |
| [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2/tree/990b2711a36180dd19d9c94b8627844866f8982a) | `990b2711a36180dd19d9c94b8627844866f8982a` | CC BY 4.0 | Independent external evaluation |

`manifests/helpsteer2_external_pairs.csv` is a derived split contract. It
contains source-row indices, hashes, binary preference labels, and score
differences, but no dataset text. The original rows must be obtained from
HelpSteer2.

## Models

| Resource | Pinned revision | License | Use |
|---|---|---|---|
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Apache-2.0 | Controlled scalar reward learner |
| [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/060db6499f32faf8b98477b0a26969ef7d8b9987) | `060db6499f32faf8b98477b0a26969ef7d8b9987` | Apache-2.0 | Observational reward-model analysis |
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf) | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` | Apache-2.0 | Frozen semantic encoder |

Python dependencies are installed from their upstream distributions and are
not vendored. Their licenses remain governed by those distributions.

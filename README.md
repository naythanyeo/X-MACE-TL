# X-MACE-TL

X-MACE-TL is a fork of [X-MACE](https://github.com/rhyan10/X-MACE), focused on flexible training, transfer learning, and multi-head excited-state models. This README describes the additions and changes made in this fork.


## What is new

- **Class-based training and testing:** `AtomDataLoaderBuilder`, `initialise_autoencoder()`, `Trainer`, and `Tester` provide reusable interfaces outside the original CLI.
- **Separate autoencoder heads:** each head groups its energy encoder, decoder, and readouts, allowing strategies to copy and configure complete heads while sharing the graph backbone.
- **Transfer-learning strategies:** naive fine-tuning, freezing, LoRA, independent heads, and cumulative correction heads. Adapted from the TL strategies implemented in the original MACE repository.
- **Multi-head data loading:** labelled datasets, per-head E0 baselines, complete geometry batches grouped by head, and optional head balancing. Dataset heads identify tasks or fidelities, rather than electronic states.
- **Energy and NAC outputs:** matrix or direct energy decoders, explicit raw and smooth NAC quantities, smooth-NAC supervision, and reporting of both raw and smooth errors.


## Upstream issues and local changes

| Upstream issue | Local behaviour |
|---|---|
| [#3 — NACs Second Interaction Layer](https://github.com/rhyan10/X-MACE/issues/3) | The autoencoder model retains vector channels through the final interaction/product stage so it can contribute to NAC predictions. Energy readouts remain invariant. |
| [#5 — Encoder Latent Dimensions](https://github.com/rhyan10/X-MACE/issues/5) | Documents the historical hidden width of 16 versus the intended 128. The current local encoder still defaults to 16; the change to 128 is not implemented here. Hidden width is separate from latent dimension. |
| [#6 — NACs Phase Consistency](https://github.com/rhyan10/X-MACE/issues/6) | NAC alignment uses consistent electronic-state phase assignments per geometry, rather than allowing independent atom/pair sign choices. |

## Set up

```bash
git clone https://github.com/naythanyeo/X-MACE-TL.git
cd X-MACE-TL

conda create -n xmace-tl-env python=3.13 pip
conda activate xmace-tl-env
python -m pip install -e .
```

## Scripts and tutorials

[X-MACE-tools](https://github.com/yoghurtfork/X-MACE-tools) contains the companion workflows:

- [JSON-driven scripts](https://github.com/yoghurtfork/X-MACE-tools/tree/main/xmace_scripts) for base training, transfer learning, and testing.
- [Tutorials](https://github.com/yoghurtfork/X-MACE-tools/tree/main/tutorials) explaining the model, classes, and strategies.
- [Data preparation](https://github.com/yoghurtfork/X-MACE-tools/tree/main/data_helpers) and [analysis utilities](https://github.com/yoghurtfork/X-MACE-tools/tree/main/analysis_tools).

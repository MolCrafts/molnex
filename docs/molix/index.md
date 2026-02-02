# Getting Started with Molix

Molix is the training backbone of the MolNex ecosystem. It provides a clean, modular engine to train your Pytorch models without boilerplate.

## Start Here

If you are new to Molix, we recommend the following learning path:

1.  [**Quickstart**](quickstart.md): Train your first model in 2 minutes.
2.  [**Data Pipelines**](data/datamodules.md): Learn how to handle molecular data with `AtomTD`.
3.  [**The Trainer**](core/trainer.md): Understand the core training loop.
4.  [**Hooks**](core/hooks.md): Master the plugin system (checkpointing, logging).

## Core Philosophy

### Decoupled Training
We believe that **models** (the math) and **training** (the loop) should be separate. Molix allows you to swap logging backends, change optimizer strategies, or move to multi-GPU training without touching your model code.

### "Zero Overhead"
Molix is designed to be as fast as raw PyTorch. The hooks system compiles away when unused, ensuring that your research code scales from laptop to cluster seamlessly.

## Architecture

*   **Trainer**: The orchestrator.
*   **State**: The source of truth for the current epoch/step.
*   **Hooks**: Event listeners for customization.

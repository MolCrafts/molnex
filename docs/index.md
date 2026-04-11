# MolNex Documentation

## Molecular Machine Learning Framework

MolNex is a modular framework for building and training molecular machine learning models. It unifies training, representation learning, and potential energy surface modeling into a cohesive ecosystem.

## The MolNex Ecosystem

MolNex is designed as a collection of specialized libraries that work together to solve molecular ML problems. Unlike monolithic frameworks, you can pick and choose the pieces you need.

It consists of four primary components:

<div class="grid cards" markdown>

-   :material-school: **molix**

    ---

    **Training Framework**
    
    A clean, component-based training system with a powerful hook mechanism.
    
    [:octicons-arrow-right-24: Learn more](molix/index.md)

-   :material-molecule: **molrep**

    ---

    **Representations**
    
    Embeddings and interactions for turning molecular graphs into semantic vectors.
    
    [:octicons-arrow-right-24: Learn more](molrep/index.md)

-   :material-atom: **molpot**

    ---

    **Potentials**
    
    Building blocks for state-of-the-art machine learning potentials.
    
    [:octicons-arrow-right-24: Learn more](molpot/index.md)

-   :material-robot: **molzoo**

    ---

    **Encoder Zoo**
    
    Pre-built encoder architectures (MACE, Allegro) ready to use.

</div>

## Getting Started

If you are new to MolNex, we recommend starting with the guides below. They will walk you through setting up your environment and training your first physics-aware model.

<div class="grid cards" markdown>

-   :material-rocket-launch: **Installation**

    ---

    Get MolNex installed on your machine.
    
    [:octicons-arrow-right-24: Install MolNex](get-started/installation.md)

-   :material-book-open-variant: **Quick Start**

    ---

    Train a physics-aware model in 5 minutes.
    
    [:octicons-arrow-right-24: Start Training](get-started/quick-start.md)

</div>

## License

BSD 3-Clause License. See [LICENSE](https://github.com/molcrafts/molnex/blob/master/LICENSE) for details.

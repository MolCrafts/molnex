# MolPot

MolPot is the composition and potentials layer of MolNex.

Its purpose is to give specialized modeling components their own place in the architecture, instead of forcing every downstream problem into the same abstraction.

## Design Role

MolPot exists because some molecular tasks need more structure in how outputs, parameters, and model components are assembled. Those concerns deserve their own layer, with language that matches the problem rather than being squeezed into one generic API.

It is one module inside a general framework, not the definition of the framework itself.

## What MolPot Optimizes For

- a natural home for composable downstream modeling
- composability across learned components and specialized terms
- clearer expression of modeling assumptions
- support for potential-based work without making it the center of the project

# Polyhedral Geometry of Time-to-First-Spike Neural Networks

This repository contains the code used to reproduce the experiments in *Polyhedral Geometry of Time-to-First-Spike Neural Networks*. 

The repository provides common implementations of the TTFS-based SNN and ReLU models, the initialization schemes considered in the paper, trajectory-based region counting, training routines, and exact two-dimensional region enumeration. Experiment-specific notebooks use these shared implementations to reproduce the numerical results and figures reported in the paper.

## Requirements

Install the required dependencies and the local package with

```bash
pip install -r requirements.txt
pip install -e .
```

## Repository layout

```text
src/ttfs_regions/
    models.py
        TTFS and ReLU architectures and TTFS causal forward passes.

    initializations.py
        Initialization schemes used in the paper, including Init 1--4
        and He/Kaiming initialization for ReLU networks.

    data.py
        MNIST and CIFAR-10 loading and preprocessing, 
        and sampling of input pairs used in the trajectory experiments.

    trajectories.py
        Trajectory-based region estimator using TTFS causal patterns and
        ReLU activation patterns.

    exact2d.py
        Polyhedral routines for exact enumeration of linear regions on
        bounded two-dimensional affine slices and extraction of exact
        classification decision boundaries.

    runners.py
        Shared training and evaluation routines for TTFS and ReLU networks.

    plotting.py
        Common plotting utilities used to generate the figures.

    configs.py
        Experimental configurations, including widths, depth schedules,
        random seeds, trajectory resolution, initialization parameters,
        and training defaults.

    utils.py
        Shared utility functions.


notebooks/
    01_ttfs_width_depth.ipynb
        TTFS width and depth-scaling experiments. Computes causal-region
        counts along fixed input trajectories for the width and depth
        configurations considered in the paper.

    02_relu_width_depth.ipynb
        Matched ReLU width and depth-scaling experiments using He
        initialization and the same trajectory protocol.

    03_shared_width_depth.ipynb
        TTFS shared weight experiments, where neurons within a layer use
        the shared incoming weight construction considered in the paper.

    04_signed_width_depth.ipynb
        Arbitrary sign TTFS experiments corresponding to Init 4.

    05_training_regions_accuracy.ipynb
        Depth-one training experiments for TTFS and matched ReLU networks,
        including region complexity during training and classification
        accuracy.

    06_exact2d.ipynb
        Exact polyhedral enumeration of TTFS and ReLU regions on bounded
        two-dimensional affine slices, together with classification
        decision boundaries.
```

The notebooks contain only experiment-specific configuration, execution, and
visualization. Model definitions, initialization schemes, data loading,
training routines, and region-counting algorithms are imported from the common
implementation in `src/ttfs_regions/`.

## Data and output directories

MNIST and CIFAR-10 are downloaded automatically to `./data` unless a different
data directory is specified in the experiment configuration.

Experimental outputs are written to `./results`. 

## Full and fast configurations

The full paper-scale experiments can be computationally expensive. The
experiment notebooks therefore also provide a fast mode that reduces the
number of seeds, architectures, trajectories, and/or trajectory samples.

Fast mode is intended only as a quick end-to-end check of the installation and
experimental pipeline. It should **not** be used to reproduce the numerical
results reported in the paper.

To reproduce the paper results, disable fast mode and use the default
paper-scale configuration specified in each notebook.

## Initialization mapping

The paper refers to the SNN initialization schemes as Init 1--4. In the code,
descriptive names are used to make the role of each initialization explicit.

- **Init 1 / `generic_positive`**  
  Positive lognormal TTFS weights with log-standard deviation `0.5` and
  thresholds sampled from `LogUniform(0.25, 4)`. No fixed neuron-wise
  time normalization is applied. This is the default positive initialization
  used in the main region experiments.

- **Init 2 / `max_regions`**  
  Positive lognormal weights with log-standard deviation `1.5`, thresholds
  calibrated near causal-prefix switching conditions, and fixed neuron-wise
  affine time normalization. This initialization is used to examine the
  sensitivity of the realized region complexity to initialization.

- **Init 3 / `training_oriented`**  
  Positive normalized incoming weights together with threshold calibration
  designed for the training and accuracy experiments.

- **Init 4 / `generic_signed_gaussian`**  
  Zero-mean Gaussian arbitrary sign weights. 

- **ReLU / `he`**  
  He/Kaiming-normal weight initialization with nonzero Gaussian biases.




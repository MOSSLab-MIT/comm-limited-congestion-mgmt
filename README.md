## Communication-Limited Multi-Agent Grid Congestion Management via Differentiable Optimization
This repository contains code to reproduce the experiments in our paper
["Communication-Limited Multi-Agent Grid Congestion Management via Differentiable Optimization"](https://dl.acm.org/doi/10.1145/3744255.3811712).

## Abstract
<p style="text-align: justify;">While there have been many recent advances in multi-agent grid control, they often rely on more communication infrastructure than is available in existing real-world transmission systems. To address this, we present a novel framework for multi-agent grid congestion management inspired by the communication architecture used by RTE, the French transmission system operator. Our framework considers communication-limited architectures in which a central coordinator is only able to periodically provide information to an ensemble of local optimization-based controllers; these local controllers do not communicate, but are nonetheless dynamically coupled. The goal of the coordinator is to provide signals that limit adverse interactions between local controllers, and to ensure that the system as a whole minimizes operational costs and violations of thermal limits. To do so, we leverage the structure of the congestion management problem to expose a parameterized family of locally-verifiable constraints for each controller that jointly imply global constraint satisfaction. The central coordinator then assigns constraint values from this family to the local controllers, with the goal of minimizing the resultant cost over the joint closed-loop trajectory under forecasted scenarios of future disturbances. We frame this as a bi-level problem, and use recent advances in differentiable optimization to find an approximate solution. We demonstrate our method on an IEEE 118-node system partitioned into three control areas. Our results demonstrate that our approach significantly lowers cost compared to RTE’s existing baseline. Furthermore, despite the significant communication constraints, our framework achieves costs comparable to the optimal cost under perfect information of future disturbances and with no restrictions on communication.

##
If you find this repository helpful in your publications, please consider citing our paper.
```bash
@inproceedings{10.1145/3744255.3811712,
author = {Chen, James Y and Drobot, St\'{e}phane and Saludjian, Lucas and Panciatici, Patrick and Chen, Pin-Yu and Jadbabaie, Ali and Donti, Priya L},
title = {Communication-Limited Multi-Agent Grid Congestion Management via Differentiable Optimization},
year = {2026},
isbn = {9798400720116},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3744255.3811712},
doi = {10.1145/3744255.3811712},
booktitle = {Proceedings of the 17th ACM International Conference on Future and Sustainable Energy Systems},
pages = {486–503},
numpages = {18},
keywords = {multi-agent control, differentiable optimization, congestion management, limited communication control},
location = {
},
series = {E-Energy '26}
}
```

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Generation
Forecast and test scenarios can be generated in the notebook ```data/scenario_generation/generate_data.ipynb```. Grid topologies and parameters are taken from the MATPOWER IEEE 118 bus test case; the MATPOWER test case does not provide line limits, so we use the values in the NREL 118 bus test case.

### Running Experiments and Baselines
Our methods $\mathsf{OURS}$ and $\mathsf{PROXY}$ can be run via ```python experiments/run_ours.py``` and ```python experiments/run_proxy.py``` respectively. These `train' on the forecast disturbances passed to them to get line limits, and then tests these line limits on the test scenarios. These take the following keyword arguments:

* Arguments that determine which set of forecast/test scenarios are used
  * `--noise_mag` controls magnitude of disturbances
  * `--offset` changes shape of disturbances
  * `--forecast_seed` determines which particular set of forecast scenarios are used for training
  * `--radius` controls size of distributions that forecast/test scenarios are drawn from
* Arguments that determine optimization parameters
  * `--epochs`
  * `--lr` 
  * `--optimizer_type` default clipped_gd
  * `--lr_schedule` default plateau
  * `--lr_decay_step`
  * `--patience` parameter for plateau lr schedule
  * `--batch_size`
  * `--max_grad_norm`

Similarly, our two baselines $\mathsf{OPT}$ and $\mathsf{DEC}$ can be run via ```python experiments/run_opt.py``` and ```python experiments/run_dec.py``` respectively. These take the following keyword arguments:
  * `--noise_mag` controls magnitude of disturbances
  * `--offset` changes shape of disturbances
  * `--radius` controls size of distributions that forecast/test scenarios are drawn from
  * `--test_skew_mag` controls magnitude of distribution shift between forecast/test scenarios

### Analysis
Results can be analyzed in ```results/generate_figures.ipynb```, which generates all the figures in the main body of the paper. To simulate examples of line limits from particular forecast/test scenarios, use ```results/line_limit_examples.ipynb```.

## Contact
Authors can be contacted at ```jamesyc@mit.edu```. We strongly encourage questions and reports of errata.
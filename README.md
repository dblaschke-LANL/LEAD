# LEAD

LEArning Damage (LEAD) is a suite of scripts to enable rapid estimation of model parameters for damage models for ductile materials using
machine learning. In its current version, the only supported damage model is TEPLA, though others are planned.
The data
we train on are comprised of free surface velocity data as well as porosity data. For example, we include files to postprocess
and filter the raw (training) data (generated, i.e. calculated or measured, outside of LEAD) and machine learning code to
predict model parameters for the damage model under consideration.
Some details about our method can be found in the paper </br>
D. N. Blaschke, T. Nguyen, M. Nitol, D. O'Malley, and S. Fensin, "Machine learning based approach to predict ductile damage model parameters for polycrystalline metals",
[Comput. Mater. Sci. 229 (2023) 112382](https://doi.org/10.1016/j.commatsci.2023.112382) ([arxiv.org/abs/2301.07790](https://arxiv.org/abs/2301.07790)).

## Authors (in alphabetical order)

Daniel N. Blaschke, Mashroor Nitol

## License
LEAD is distributed according to the [LICENSE](LICENSE) file. All contributions made by employees of Los Alamos National Laboratory are governed by that license.


Copyright (c) 2023, Triad National Security, LLC. All rights reserved.

The LANL development team asks that any forks or derivative works include appropriate attribution and citation of the LANL development team's original work.


## Requirements

* Python >=3.8,</br>
* [tensorflow](https://www.tensorflow.org),</br>
* [keras](https://keras.io),</br>
* [scikit-learn](https://scikit-learn.org),</br>
* [numpy](https://docs.scipy.org/doc/numpy/user/),</br>
* [scipy](https://docs.scipy.org/doc/scipy/reference/),</br>
* [pandas](https://pandas.pydata.org),</br>
* [natsort](https://natsort.readthedocs.io),</br>
* [matplotlib](https://matplotlib.org)</br>


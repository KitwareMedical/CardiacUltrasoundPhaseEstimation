CardiacUltrasoundPhaseEstimation
=================================

This repository contains an image-based method for cardio-respiratory phase
estimation, gating, and temporal super-resolution of cardiac ultrasound.

The core algorithm is implemented in the class [uspgs.USPGS](https://github.com/KitwareMedical/CardiacUltrasoundPhaseEstimation/blob/master/uspgs.py#L22). 
Please see the docstrings to understand its functionality.

There are two jupyter notebooks that will further help understand how to use the method:
* [MICCAI_2017_experiments.ipynb](https://github.com/KitwareMedical/CardiacUltrasoundPhaseEstimation/blob/master/MICCAI_2017_experiments.ipynb) - contains the code used to generate the results for the MICCAI submission.
* [WinProbe-Experiments.ipynb](https://github.com/KitwareMedical/CardiacUltrasoundPhaseEstimation/blob/master/WinProbe-Experiments.ipynb) - shows how to apply the algorithm on WinProbe data.

Prerequisites
-------------

The following python packages need to be installed in your virtual environment 
to be able to run the code: 

* numpy
* scipy
* scikit-learn
* statsmodels
* matplotlib
* jupyter
* angles
* SimpleITK
* MedPy
* opencv-python

You can run the following command to install these dependencies:

```
pip install requirements.txt
```

The code uses opencv's Python interface for reading, displaying, and writing videos. 

pip installing the requirements using the aforementioned command attempts to 
install opencv's python interface by installing the `opencv-python>=3.2.07` 
package from [PyPI](https://pypi.python.org/pypi/opencv-python) for convenience. 

However since this is not an official release of opencv, it may cause issues,
especially on OSX and Linux platforms. If this happens, download and install 
opencv binaries for your operating system from the official opencv website
 [here](http://opencv.org/releases.html). 


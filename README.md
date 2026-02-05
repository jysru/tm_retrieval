# tm_retrieval

(True) Optical field Transmission Matrix retrieval in Pytorch + Lightning.

This my open implementation of the method described in our paper: https://ieeexplore.ieee.org/document/10459341.

An example is given from a simulated Step-Index fiber dataset.

Tweak it to your own needs, but please cite the original paper!

Thanks, and happy retrieval!

## Installation

- Install Python >= 3.11

- Install Pytorch (with NVIDIA GPU support) into your environment:
```bash
cd <this_project>
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

- Install this project (in editable mode if you want to make changes) into your environment:
```bash
cd <this_project>
pip install -e .
```

- Download the example data file to get along with the provided example from here: https://drive.proton.me/urls/VV000BHHNR#XVWnglrm2UXh, then put it into the `./resources` folder

## Example

Check [pixel_to_pixel_fiber_transmission_matrix.ipynb](./examples/pixel_to_pixel_fiber_transmission_matrix.ipynb)


## Tested

With Python 3.13.5, Pytorch 2.10.0+cu130, lightning 2.6.1
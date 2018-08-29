# CNN-PS
# Project Title

Satoshi Ikeahta. CNN-PS: CNN-based Photometric Stereo for General Non-Convex Surfaces, ECCV2018.


## Getting Started

This is a Keras implementation of a CNN for estimating surface normals from images captured under different illumination.

### Prerequisites

- Python3.5+
- Keras2.0+
- numpy
- OpenCV3

Tested on:
- Ubuntu 16.04, Python 3.5.2, Keras 2.0.3, Tensorflow(-gpu) 1.0.1, Theano 0.9.0, CUDA 8.0, cuDNN 5.0
  - CPU: Intel® Xeon(R) CPU E5-1650 v4 @ 3.60GHz × 12 , GPU: 3x GeForce GTX1080Ti, Memory 64GB

### Running the tests
For testing network (with DiLiGenT dataset)

```
python test.py
```
The pretrained model for TensorFlow backend is included (weight_and_model.hdf5)

## Running the training
I will prepare for the training data soon...

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
This work was supported by JSPS KAKENHI Grant Number JP17H07324.

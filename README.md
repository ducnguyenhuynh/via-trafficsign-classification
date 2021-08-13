# Traffic sign classification

## Training

- Using LeNet: [lenet/Tutorial.ipynb](lenet/Tutorial.ipynb).
- Using ResNet: [resnet/Tutorial.ipynb](resnet/Tutorial.ipynb).

## Inference with ONNX

- Convert your model to ONNX:
    + Using LeNet: [lenet/convert2onnx.py](lenet/convert2onnx.py).
    + Using ResNet: [resnet/convert2onnx.py](resnet/convert2onnx.py).

Or download pretrained models:

```
cd inference-onnx
sh download_models.sh
```

- Compile code:

```
cd inference-onnx
mkdir build
cd build
cmake ..
make
```

- Run:

```
./traffic_sign_classifier <model-path> <label-file> <image-path>
```

For example:

```
./traffic_sign_classifier ../models/resnet18.onnx ../models/classes.txt ../sample_data/00003.jpg
```

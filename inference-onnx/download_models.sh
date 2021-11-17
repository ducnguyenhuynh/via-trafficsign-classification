!/bin/sh

mkdir models > null
wget -O models/resnet18.onnx https://github.com/ducnguyenhuynh/via-trafficsign-classification/releases/download/via-tf-dataset/resnet18.onnx
wget -O models/lenet.onnx https://github.com/ducnguyenhuynh/via-trafficsign-classification/releases/download/via-tf-dataset/lenet.onnx
wget -O models/resnet18_64x64.onnx https://github.com/ducnguyenhuynh/via-trafficsign-classification/releases/download/via-tf-dataset/resnet18_64x64.onnx

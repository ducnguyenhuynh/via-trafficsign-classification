import torch
import torch.nn as nn
# from models import LeNet


# model = LeNet()

model = nn.Sequential(nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                    nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 7)).to('cuda')

# model.load_state_dict(torch.load("tut5-model.pt")).to('cuda')



x = torch.randn(1, 3, 28, 28).to('cuda')
out = model(x)

torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "lenet.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                #                 'output' : {0 : 'batch_size'}})
)


# import onnx

# onnx_model = onnx.load("lenet.onnx")
# onnx.checker.check_model(onnx_model)
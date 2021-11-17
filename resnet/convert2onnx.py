import torch
from models import *
from dataloader import Dataloader


model = ResNet(resnet18_config, 7).to('cuda')
model.load_state_dict(torch.load("tut5-model.pt"))
print(model)


x = torch.randn(1, 3, 128, 128).to('cuda')
out = model(x)

torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "resnet18.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

# dataloader = Dataloader("../dataset", 128)

# train_data, valid_data, test_data =  dataloader.get_dataset()

# BATCH_SIZE = 64

# train_iterator = data.DataLoader(train_data, 
#                                 shuffle = True, 
#                                 batch_size = BATCH_SIZE)

# valid_iterator = data.DataLoader(valid_data, 
#                                 batch_size = BATCH_SIZE)

# test_iterator = data.DataLoader(test_data, 
#                                 batch_size = BATCH_SIZE)



# import onnx

# onnx_model = onnx.load("resnet18.onnx")




import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model-path', default = "models/lenet.onnx", required = True)
parser.add_argument('--label-file', default = "models/classes.txt", required = True)
parser.add_argument('--image-path', default = "../dataset/test/0/00059.jpg", required = True)


# Hàm softmax dùng để chuyển đổi đầu ra của mô hình về dạng xác suất
def softmax(inputs):
    sum = 0
    result =  []
    for i in inputs:
        sum+=np.exp(i)
    for i in inputs:
        result.append(np.exp(i)/sum)
    return np.array(result)


def read_label(label_file):
    with open(label_file,"r") as file:
        labels = file.readlines()
        # print(labels)
    return labels

if __name__ == "__main__":

    args = parser.parse_args()

    mean= [0.4151, 0.3771, 0.4568]
    std = [0.2011, 0.2108, 0.1896]
    # print(args.model_path)
        
    net =  cv2.dnn.readNetFromONNX(args.model_path)

    image = cv2.imread(args.image_path)
    labels = read_label(args.label_file)

    if image is None:
        print("Image path wrong!!!!")

    else:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        # input shape of model lenet is (28,28,3), if you use resnet, needed to change size to (128,128) or (64,64)
        image = cv2.resize(image, (64,64))
        image = (image/255 - mean)/std
        image = np.transpose(image,(2,0,1))    
        image = np.expand_dims(image, axis =0)

        net.setInput(image)
        preds = net.forward()

        prediction = softmax(preds[0])
        cls = prediction.argmax()
        score = prediction[cls]
        
        print(labels[cls].strip() + ": ", score)
        print("Done")
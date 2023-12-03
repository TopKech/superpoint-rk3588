import numpy as np
import cv2
from rknn.api import RKNN
import torch
from superpointnet import SuperPointNet, postprocess


if __name__ == "__main__":
    traced = torch.jit.load("traced.pt")

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    input_size_list = [[1, 1, 512, 512]]
    rknn.config(mean_values=[0,], std_values=[255], target_platform="rk3588")
    ret = rknn.load_pytorch(model="traced.pt", input_size_list=input_size_list)
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    ret = rknn.init_runtime()
    print('done')
    # Set inputs
    img = cv2.imread('./example.jpg', cv2.IMREAD_UNCHANGED)[np.newaxis, np.newaxis,...]

    outputs = rknn.inference(inputs=[img], data_format="nchw")
    rknn.release()

    outputs_torch = traced(torch.Tensor(img)/255)
    print(np.abs(outputs_torch[0].detach().numpy()-outputs[0]).max())

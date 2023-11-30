from rknn.api import RKNN
import torch
from superpointnet import SuperPointNet


if __name__ == '__main__':
    model = "./superPointNet_114000_checkpoint.pth.tar"
    traced = "./traced.pt"
    sp = SuperPointNet(model)
    sp.eval()
    traced_sp = torch.jit.trace(sp, torch.Tensor(1, 1, 512, 512))
    traced_sp.save(traced)

    input_size_list = [[1, 1, 512, 512]]

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[0,], std_values=[255], target_platform="rk3588")
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=traced, input_size_list=input_size_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn('./sp.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    rknn.release()

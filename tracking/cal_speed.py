import onnxruntime
import onnx
import time
import torch
import numpy as np

def get_data(bs=1, sz_x=256, sz_z=128):
    img_x = torch.randn(bs, 3, sz_x, sz_x, requires_grad=True)
    img_z = torch.randn(bs, 3, sz_z, sz_z, requires_grad=True)
    return img_x, img_z

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    for i in range(24):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1 + i
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        onnx_dir = "complete.onnx"
        model = onnx.load(onnx_dir)
        ort_session = onnxruntime.InferenceSession(onnx_dir, providers=['CPUExecutionProvider'], sess_options=options)

        img_x, img_z = get_data()
        ort_inputs = {'template': to_numpy(img_z), 'search': to_numpy(img_x)}

        for _ in range(50):
            ort_outs = ort_session.run(None, ort_inputs)

        time_list = []
        for _ in range(1000):
            start = time.time()
            ort_outs = ort_session.run(None, ort_inputs)
            end = time.time()
            time_list.append(end - start)
        print("线程数：", i+1)
        print("最小时间：", min(time_list))
        print("最高FPS", 1 / min(time_list))
        print("平均时间：", sum(time_list) / 1000)
        print("平均FPS", 1000 / sum(time_list))
        time.sleep(5)


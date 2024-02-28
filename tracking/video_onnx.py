import onnx
import onnxruntime
import argparse
import cv2 as cv
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    # parser.add_argument('videofile', type=str, help='path to a video file.')
    # args = parser.parse_args()
    #
    onnx_dir = "complete.onnx"
    model = onnx.load(onnx_dir)
    ort_session = onnxruntime.InferenceSession(onnx_dir, providers=['CPUExecutionProvider'])
    #
    # cap = cv.VideoCapture(args.videofile)
    # display_name = 'Display: '
    # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    # cv.resizeWindow(display_name, 960, 720)
    # success, frame = cap.read()
    # cv.imshow(display_name, frame)
    #
    # frame_disp = frame.copy()
    # cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 1)
    # x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
    #
    pre = Preprocessor()
    # z_patch_arr, resize_factor, z_amask_arr = sample_target(frame, [x, y, w, h], 2, 128)
    # template = pre.process(z_patch_arr, z_amask_arr)
    #
    # x_patch_arr, resize_factor, x_amask_arr = sample_target(frame, [x, y, w, h], 4, 256)
    # search = pre.process(x_patch_arr, x_amask_arr)
    #
    # out_string = ("output1", "output2", "output3")
    # ort_inputs = {'template': to_numpy(template.tensors), 'search': to_numpy(search.tensors)}
    # outs = ort_session.run(out_string, ort_inputs)
    #
    # conf_map = outs[0].reshape((16, 16))
    # size_map = outs[1].reshape((2, 16, 16))
    # offset_map = outs[2].reshape((2, 16, 16))
    #
    # pos = conf_map.argmax()
    # x = pos // 16
    # y = pos % 16
    #
    # cx = x + offset_map
    # a = 1
    img = cv.imread("/storage/imagenet1k/train/n01440764/n01440764_18.JPEG")
    x1, y1, w, h = 200, 200, 50, 50
    # rect = cv.selectROI(img)
    x_patch_arr, resize_factor, x_amask_arr = sample_target(img, [x1, y1, w, h], 4,
                                                            output_sz=256)  # (x1, y1, w, h)
    search = pre.process(x_patch_arr, x_amask_arr)

    x_patch_arr, resize_factor, x_amask_arr = sample_target(img, [x1, y1, w, h], 2,
                                                            output_sz=128)  # (x1, y1, w, h)
    template = pre.process(x_patch_arr, x_amask_arr)


    out_string = ("output1", "output2", "output3")
    ort_inputs = {'template': to_numpy(template.tensors), 'search': to_numpy(search.tensors)}
    outs = ort_session.run(out_string, ort_inputs)
    a = 1


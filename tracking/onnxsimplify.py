import onnx
from onnxsim import simplify

model = onnx.load("vttrack.onnx")

model_simp, check = simplify(model)

onnx.save(model_simp, "object_tracking_vittrack_2023sep.onnx")

# onnx.save(onnx.shape_inference.infer_shapes(onnx.load("vttrack.onnx")), "vttrack.onnx")

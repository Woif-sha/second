import onnx
from onnxsim import simplify

model = onnx.load(r'E:\PyCharm Project\CrowdRecognition\weights\yolov5x_cartoon.onnx')
simplified_model, check = simplify(model)
onnx.save(simplified_model, r'E:\PyCharm Project\CrowdRecognition\weights\yolov5x_cartoon_simplify.onnx')
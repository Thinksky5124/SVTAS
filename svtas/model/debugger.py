'''
Author       : Thyssen Wen
Date         : 2022-09-24 16:46:51
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 20:45:30
Description  : Debugger Class for Infer debugging
FilePath     : /SVTAS/svtas/model/debugger.py
'''
import torch 
import onnx 
import onnxruntime 
import numpy as np
from ..utils.logger import get_logger
 
class DebugOp(torch.autograd.Function): 
    @staticmethod 
    def forward(ctx, x, name): 
        return x 
 
    @staticmethod 
    def symbolic(g, x, name): 
        return g.op("my::Debug", x, name_s=name) 
 
debug_apply = DebugOp.apply 
 
class Debugger(): 
    def __init__(self): 
        super().__init__() 
        self.torch_value = dict() 
        self.onnx_value = dict() 
        self.output_debug_name = []
        self.logger = None
 
    def debug(self, x, name): 
        self.torch_value[name] = x.detach().cpu().numpy() 
        return debug_apply(x, name) 
 
    def extract_debug_model(self, input_path, output_path): 
        model = onnx.load(input_path) 
        inputs = [input.name for input in model.graph.input] 
        outputs = [output.name for output in model.graph.output] 
 
        for node in model.graph.node: 
            if node.op_type == 'Debug': 
                debug_name = node.attribute[0].s.decode('ASCII') 
                self.output_debug_name.append(debug_name) 
 
                output_name = node.output[0] 
                outputs.append(output_name) 
 
                node.op_type = 'Identity' 
                node.domain = '' 
                del node.attribute[:] 
        e = onnx.utils.Extractor(model) 
        extracted = e.extract_model(inputs, outputs) 
        onnx.save(extracted, output_path) 

    def run_debug_model(self, input, debug_model): 
        sess = onnxruntime.InferenceSession(debug_model) 
 
        onnx_outputs = sess.run(None, input) 
        for name, value in zip(self.output_debug_name, onnx_outputs[1:]): 
            self.onnx_value[name] = value 
 
    def print_debug_result(self): 
        for name in self.torch_value.keys(): 
            if name in self.onnx_value: 
                mse = np.mean(self.torch_value[name] - self.onnx_value[name])**2
                self.logger.info(f"{name} MSE: {mse}")
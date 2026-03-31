"""
模型导出模块
支持 ONNX、TorchScript 等格式
"""
import torch
import torch.onnx
from typing import Dict, Tuple, Optional
import os


class ModelExporter:
    """
    模型导出器
    支持多种导出格式
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        初始化导出器
        
        Args:
            model: PyTorch 模型
            device: 设备
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def export_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, int] = (1, 512),
        opset_version: int = 14,
        dynamic_axes: Optional[Dict] = None
    ):
        """
        导出 ONNX 格式
        
        Args:
            output_path: 输出路径
            input_shape: 输入形状 (batch_size, seq_len)
            opset_version: ONNX 算子集版本
            dynamic_axes: 动态轴配置
        """
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        
        # 创建示例输入
        dummy_input = torch.randint(
            0, 1000, input_shape,
            dtype=torch.long, device=self.device
        )
        
        # 导出
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        print(f"ONNX model exported to: {output_path}")
        
        # 验证模型
        self._verify_onnx(output_path, dummy_input)
    
    def _verify_onnx(self, onnx_path: str, dummy_input: torch.Tensor):
        """
        验证导出的 ONNX 模型
        
        Args:
            onnx_path: ONNX 模型路径
            dummy_input: 示例输入
        """
        try:
            import onnx
            import onnxruntime as ort
            
            # 检查模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # 使用 ONNX Runtime 验证
            ort_session = ort.InferenceSession(onnx_path)
            
            # 准备输入
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            
            # 运行推理
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # PyTorch 推理
            with torch.no_grad():
                pytorch_output = self.model(dummy_input).cpu().numpy()
            
            # 比较输出
            diff = abs(ort_outputs[0] - pytorch_output).max()
            
            if diff < 1e-5:
                print(f"✓ ONNX verification passed! Max diff: {diff:.2e}")
            else:
                print(f"⚠ ONNX verification warning: Max diff: {diff:.2e}")
        
        except ImportError:
            print("⚠ onnx/onnxruntime not installed, skipping verification")
        except Exception as e:
            print(f"⚠ ONNX verification failed: {e}")
    
    def export_torchscript(
        self,
        output_path: str,
        input_shape: Tuple[int, int] = (1, 512),
        method: str = 'trace'
    ):
        """
        导出 TorchScript 格式
        
        Args:
            output_path: 输出路径
            input_shape: 输入形状
            method: 导出方法 ('trace' 或 'script')
        """
        dummy_input = torch.randint(
            0, 1000, input_shape,
            dtype=torch.long, device=self.device
        )
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        if method == 'trace':
            # 使用 tracing
            traced_model = torch.jit.trace(self.model, dummy_input)
            traced_model.save(output_path)
        else:
            # 使用 scripting
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(output_path)
        
        print(f"TorchScript model exported to: {output_path}")
    
    def export_checkpoint(
        self,
        output_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        导出训练检查点
        
        Args:
            output_path: 输出路径
            metadata: 额外元数据
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {},
            'metadata': metadata or {}
        }
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        torch.save(checkpoint, output_path)
        
        print(f"Checkpoint exported to: {output_path}")
    
    def export_for_mobile(
        self,
        output_path: str,
        input_shape: Tuple[int, int] = (1, 512),
        backend: str = 'lite_interp'
    ):
        """
        导出移动端模型
        
        Args:
            output_path: 输出路径
            input_shape: 输入形状
            backend: 后端类型
        """
        dummy_input = torch.randint(
            0, 1000, input_shape,
            dtype=torch.long, device=self.device
        )
        
        # 转换为 TorchScript
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        # 优化模型
        optimized_model = torch.jit.optimize_for_inference(traced_model)
        
        # 保存
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        optimized_model._save_for_lite_interpreter(output_path)
        
        print(f"Mobile model exported to: {output_path}")


class ONNXInference:
    """
    ONNX 模型推理器
    """
    
    def __init__(self, onnx_path: str, providers: Optional[list] = None):
        """
        初始化 ONNX 推理器
        
        Args:
            onnx_path: ONNX 模型路径
            providers: 执行提供者列表
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX inference")
        
        if providers is None:
            # 自动选择最佳 provider
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        推理
        
        Args:
            input_ids: 输入张量
        
        Returns:
            输出张量
        """
        # 转换为 numpy
        input_numpy = input_ids.cpu().numpy()
        
        # 运行推理
        outputs = self.session.run(
            None,
            {self.input_name: input_numpy}
        )
        
        # 转换回 tensor
        return torch.from_numpy(outputs[0])
    
    def benchmark(self, input_shape: tuple, num_runs: int = 100) -> Dict:
        """
        性能测试
        
        Args:
            input_shape: 输入形状
            num_runs: 运行次数
        
        Returns:
            性能指标
        """
        import time
        import numpy as np
        
        dummy_input = np.random.randint(0, 1000, input_shape, dtype=np.int64)
        
        # 预热
        for _ in range(10):
            _ = self.session.run(None, {self.input_name: dummy_input})
        
        # 测试
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.session.run(None, {self.input_name: dummy_input})
            times.append(time.time() - start)
        
        return {
            'avg_latency_ms': sum(times) / len(times) * 1000,
            'min_latency_ms': min(times) * 1000,
            'max_latency_ms': max(times) * 1000,
            'throughput_samples_per_sec': num_runs / sum(times)
        }


def export_model_pipeline(
    model: torch.nn.Module,
    output_dir: str,
    input_shape: Tuple[int, int] = (1, 512),
    formats: list = ['onnx', 'torchscript', 'checkpoint']
):
    """
    一键导出模型到多种格式
    
    Args:
        model: PyTorch 模型
        output_dir: 输出目录
        input_shape: 输入形状
        formats: 导出格式列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    exporter = ModelExporter(model)
    
    if 'onnx' in formats:
        onnx_path = os.path.join(output_dir, 'model.onnx')
        exporter.export_onnx(onnx_path, input_shape)
    
    if 'torchscript' in formats:
        ts_path = os.path.join(output_dir, 'model.pt')
        exporter.export_torchscript(ts_path, input_shape)
    
    if 'checkpoint' in formats:
        ckpt_path = os.path.join(output_dir, 'model.ckpt')
        exporter.export_checkpoint(ckpt_path)
    
    print(f"\nAll models exported to: {output_dir}")

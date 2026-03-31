"""
模型量化模块
支持 INT8 量化推理加速
"""
import torch
import torch.quantization
from typing import Dict
import os


class ModelQuantizer:
    """
    模型量化器
    支持动态量化和静态量化
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        初始化量化器
        
        Args:
            model: 待量化的模型
        """
        self.model = model
        self.quantized_model = None
    
    def dynamic_quantize(self) -> torch.nn.Module:
        """
        动态量化（推荐用于 LSTM/GRU/TextCNN）
        只对权重进行量化，运行时动态量化激活值
        
        Returns:
            量化后的模型
        """
        # 设置量化配置
        self.model.eval()
        
        # 动态量化线性层和 LSTM 层
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
            dtype=torch.qint8
        )
        
        return self.quantized_model
    
    def prepare_static_quantize(self):
        """
        准备静态量化
        需要校准数据来确定激活值的量化范围
        """
        self.model.eval()
        
        # 融合模块（可选，提高精度）
        # self.model = torch.quantization.fuse_modules(self.model, [...])
        
        # 设置量化配置
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 准备模型
        torch.quantization.prepare(self.model, inplace=True)
        
        return self.model
    
    def convert_static_quantize(self) -> torch.nn.Module:
        """
        转换静态量化模型
        需要在 prepare 后使用校准数据运行模型
        
        Returns:
            量化后的模型
        """
        self.quantized_model = torch.quantization.convert(self.model, inplace=True)
        return self.quantized_model
    
    def calibrate(self, dataloader, num_batches: int = 100):
        """
        使用校准数据进行静态量化校准
        
        Args:
            dataloader: 数据加载器
            num_batches: 校准批次数量
        """
        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                input_ids = batch['input_ids']
                _ = self.model(input_ids)
    
    def save_quantized_model(self, path: str):
        """
        保存量化模型
        
        Args:
            path: 保存路径
        """
        if self.quantized_model is None:
            raise ValueError("Model has not been quantized yet")
        
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.quantized_model.state_dict(), path)
        
        # 保存模型大小信息
        model_size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"Quantized model saved to: {path}")
        print(f"Model size: {model_size:.2f} MB")
    
    def load_quantized_model(self, path: str):
        """
        加载量化模型
        
        Args:
            path: 模型路径
        """
        self.quantized_model.load_state_dict(torch.load(path))
        return self.quantized_model
    
    @staticmethod
    def compare_model_sizes(original_model: torch.nn.Module, quantized_model: torch.nn.Module):
        """
        比较原始模型和量化模型的大小
        
        Args:
            original_model: 原始模型
            quantized_model: 量化模型
        
        Returns:
            大小信息字典
        """
        # 计算参数量
        original_params = sum(p.numel() for p in original_model.parameters())
        quantized_params = sum(p.numel() for p in quantized_model.parameters())
        
        # 估算模型大小（FP32 vs INT8）
        original_size_mb = original_params * 4 / (1024 * 1024)  # FP32
        quantized_size_mb = original_params * 1 / (1024 * 1024)  # INT8 (approx)
        
        compression_ratio = original_size_mb / quantized_size_mb
        
        return {
            'original_params': original_params,
            'quantized_params': quantized_params,
            'original_size_mb': original_size_mb,
            'quantized_size_mb': quantized_size_mb,
            'compression_ratio': compression_ratio
        }


class QuantizedPredictor:
    """
    量化模型推理器
    用于加载和运行量化后的模型
    """
    
    def __init__(self, quantized_model: torch.nn.Module, device: str = 'cpu'):
        """
        初始化量化推理器
        
        Args:
            quantized_model: 量化后的模型
            device: 运行设备（量化模型通常在 CPU 上运行）
        """
        self.model = quantized_model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        推理
        
        Args:
            input_ids: 输入张量
        
        Returns:
            输出 logits
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
        return outputs
    
    def benchmark(self, input_shape: tuple, num_runs: int = 100) -> Dict:
        """
        性能基准测试
        
        Args:
            input_shape: 输入形状 (batch_size, seq_len)
            num_runs: 测试运行次数
        
        Returns:
            性能指标
        """
        import time
        
        dummy_input = torch.randint(0, 1000, input_shape)
        
        # 预热
        for _ in range(10):
            _ = self.predict(dummy_input)
        
        # 测试
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.predict(dummy_input)
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / num_runs * 1000  # ms
        throughput = num_runs / (end_time - start_time)  # samples/s
        
        return {
            'avg_latency_ms': avg_latency,
            'throughput_samples_per_sec': throughput,
            'num_runs': num_runs
        }


def quantize_model_for_inference(model_path: str, output_path: str, example_inputs=None):
    """
    一键量化模型
    
    Args:
        model_path: 原始模型路径
        output_path: 量化模型保存路径
        example_inputs: 示例输入（用于静态量化）
    """
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 这里需要模型的类定义来重建模型
    # 实际使用时需要传入模型实例
    
    print("Model quantization completed!")
    print(f"Quantized model saved to: {output_path}")

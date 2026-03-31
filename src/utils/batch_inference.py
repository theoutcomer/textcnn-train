"""
批量推理优化
支持多进程并行推理
"""
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Callable, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
import time


class BatchInferenceEngine:
    """
    批量推理引擎
    支持多进程并行和 GPU 加速
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cpu',
        batch_size: int = 64,
        num_workers: int = 4,
        use_multiprocessing: bool = False
    ):
        """
        初始化推理引擎
        
        Args:
            model: 模型
            device: 设备
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
            use_multiprocessing: 是否使用多进程推理
        """
        self.model = model
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_multiprocessing = use_multiprocessing
        
        self.model.to(self.device)
        self.model.eval()
    
    def infer(
        self,
        dataloader: DataLoader,
        return_probs: bool = True
    ) -> List[Dict]:
        """
        批量推理
        
        Args:
            dataloader: 数据加载器
            return_probs: 是否返回概率
        
        Returns:
            推理结果列表
        """
        results = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inferencing"):
                input_ids = batch['input_ids'].to(self.device)
                
                # 推理
                outputs = self.model(input_ids)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                # 处理结果
                for prob in probs:
                    result = {'probabilities': prob.tolist()}
                    if not return_probs:
                        result = {'predictions': (prob > 0.5).astype(int).tolist()}
                    results.append(result)
        
        return results
    
    def infer_parallel(
        self,
        texts: List[str],
        encode_fn: Callable,
        num_processes: Optional[int] = None
    ) -> List[Dict]:
        """
        并行推理（CPU 多进程）
        
        Args:
            texts: 文本列表
            encode_fn: 编码函数
            num_processes: 进程数
        
        Returns:
            推理结果列表
        """
        if num_processes is None:
            num_processes = min(mp.cpu_count(), 4)
        
        # 分割数据
        chunk_size = len(texts) // num_processes
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # 并行推理
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(self._infer_chunk, chunk, encode_fn)
                for chunk in chunks
            ]
            
            results = []
            for future in futures:
                results.extend(future.result())
        
        return results
    
    def _infer_chunk(self, texts: List[str], encode_fn: Callable) -> List[Dict]:
        """
        推理数据块（用于多进程）
        """
        results = []
        
        # 编码
        encoded = [encode_fn(text) for text in texts]
        
        # 批处理
        for i in range(0, len(encoded), self.batch_size):
            batch = encoded[i:i + self.batch_size]
            
            # 填充
            max_len = max(len(e) for e in batch)
            padded = []
            for e in batch:
                if len(e) < max_len:
                    e = e + [0] * (max_len - len(e))
                padded.append(e)
            
            # 推理
            input_tensor = torch.tensor(padded, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()
            
            for prob in probs:
                results.append({'probabilities': prob.tolist()})
        
        return results
    
    def benchmark(
        self,
        dataloader: DataLoader,
        num_warmup: int = 10,
        num_runs: int = 100
    ) -> Dict:
        """
        性能基准测试
        
        Args:
            dataloader: 数据加载器
            num_warmup: 预热次数
            num_runs: 测试次数
        
        Returns:
            性能指标
        """
        # 预热
        print("Warming up...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_warmup:
                    break
                input_ids = batch['input_ids'].to(self.device)
                _ = self.model(input_ids)
        
        # 测试
        print(f"Benchmarking ({num_runs} runs)...")
        times = []
        total_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_runs:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                batch_size = input_ids.size(0)
                
                start = time.time()
                _ = self.model(input_ids)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end = time.time()
                
                times.append(end - start)
                total_samples += batch_size
        
        # 计算指标
        total_time = sum(times)
        avg_latency = np.mean(times) * 1000  # ms
        throughput = total_samples / total_time
        
        return {
            'device': str(self.device),
            'batch_size': self.batch_size,
            'total_samples': total_samples,
            'total_time_sec': total_time,
            'avg_latency_ms': avg_latency,
            'throughput_samples_per_sec': throughput,
            'p50_latency_ms': np.percentile(times, 50) * 1000,
            'p95_latency_ms': np.percentile(times, 95) * 1000,
            'p99_latency_ms': np.percentile(times, 99) * 1000
        }


class AsyncInferenceEngine:
    """
    异步推理引擎
    适用于高并发场景
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cpu',
        max_batch_size: int = 64,
        max_wait_ms: float = 10.0
    ):
        """
        初始化异步推理引擎
        
        Args:
            model: 模型
            device: 设备
            max_batch_size: 最大批次大小
            max_wait_ms: 最大等待时间（毫秒）
        """
        self.model = model
        self.device = torch.device(device)
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        self.model.to(self.device)
        self.model.eval()
        
        # 请求队列
        self.request_queue = []
        self.results_cache = {}
    
    async def predict_async(self, request_id: str, input_ids: torch.Tensor):
        """
        异步预测（占位实现）
        实际实现需要使用 asyncio 和队列
        
        Args:
            request_id: 请求 ID
            input_ids: 输入张量
        
        Returns:
            预测结果
        """
        # 添加到队列
        self.request_queue.append({
            'id': request_id,
            'input': input_ids
        })
        
        # 如果队列满或等待超时，执行批处理
        if len(self.request_queue) >= self.max_batch_size:
            return await self._process_batch()
        
        # 否则等待
        # ... 实际实现需要异步等待逻辑
    
    async def _process_batch(self):
        """
        处理批次请求
        """
        if not self.request_queue:
            return []
        
        # 收集输入
        batch_inputs = []
        request_ids = []
        
        for req in self.request_queue[:self.max_batch_size]:
            batch_inputs.append(req['input'])
            request_ids.append(req['id'])
        
        # 清空已处理的请求
        self.request_queue = self.request_queue[self.max_batch_size:]
        
        # 批处理
        batch_tensor = torch.stack(batch_inputs).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probs = torch.sigmoid(outputs).cpu()
        
        # 分发结果
        results = []
        for i, req_id in enumerate(request_ids):
            result = {'probabilities': probs[i].numpy().tolist()}
            self.results_cache[req_id] = result
            results.append(result)
        
        return results


def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    优化模型用于推理
    
    Args:
        model: 原始模型
    
    Returns:
        优化后的模型
    """
    model.eval()
    
    # 融合操作（如果适用）
    # model = torch.quantization.fuse_modules(model, [...])
    
    # 设置为评估模式
    model.eval()
    
    # 禁用梯度计算
    for param in model.parameters():
        param.requires_grad = False
    
    return model

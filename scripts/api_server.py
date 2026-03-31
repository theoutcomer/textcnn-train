"""FastAPI 服务部署"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pickle
import torch
import jieba
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from src.models.textcnn import TextCNN


class PredictRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.2
    return_probs: Optional[bool] = False


class PredictResponse(BaseModel):
    text: str
    labels: List[str]
    probabilities: Optional[dict] = None


class TextCNNService:
    def __init__(self, model_dir: str, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # 加载词汇表
        with open(os.path.join(model_dir, 'vocab.pkl'), 'rb') as f:
            self.vocab = pickle.load(f)
        
        # 处理词汇表（可能是 Vocabulary 对象或保存的 dict）
        if hasattr(self.vocab, 'word2idx'):
            # Vocabulary 对象
            self.vocab_dict = self.vocab.word2idx
            self.max_len = getattr(self.vocab, 'max_len', 512)
        elif isinstance(self.vocab, dict):
            # 检查是否是保存的 Vocabulary dict（包含 word2idx 键）
            if 'word2idx' in self.vocab:
                self.vocab_dict = self.vocab['word2idx']
                self.max_len = self.vocab.get('max_size', 512)
            else:
                # 普通 dict
                self.vocab_dict = self.vocab
                self.max_len = 512
        else:
            self.vocab_dict = {}
            self.max_len = 512
        
        print(f"Vocab type: {type(self.vocab)}")
        print(f"Vocab size: {len(self.vocab_dict)}")
        print(f"Max length: {self.max_len}")
        
        # 加载标签
        with open(os.path.join(model_dir, 'labels.pkl'), 'rb') as f:
            labels_data = pickle.load(f)
            self.labels = labels_data['idx2label']
        
        # 加载模型
        checkpoint = torch.load(
            os.path.join(model_dir, 'best_model.pt'),
            map_location=self.device
        )
        
        model_config = checkpoint['model_config']
        self.model = TextCNN(
            vocab_size=model_config['vocab_size'],
            embed_dim=model_config['embed_dim'],
            num_classes=model_config['num_classes'],
            filter_sizes=model_config.get('filter_sizes', [3, 4, 5]),
            num_filters=model_config.get('num_filters', 100),
            dropout=model_config.get('dropout', 0.5)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.max_len = 512
        print(f"Model loaded: {len(self.labels)} labels")
        print(f"Labels: {list(self.labels.values())}")
    
    def text_to_ids(self, text: str) -> torch.Tensor:
        """文本转换为 ID 序列"""
        tokens = list(jieba.cut(text))
        
        # 处理词汇表（可能是 dict 或 Vocabulary 对象）
        if hasattr(self.vocab, 'word2idx'):
            vocab_dict = self.vocab.word2idx
        else:
            vocab_dict = self.vocab
        
        unk_id = vocab_dict.get('<UNK>', 1)
        pad_id = vocab_dict.get('<PAD>', 0)
        
        ids = [vocab_dict.get(t, unk_id) for t in tokens]
        
        # 截断或填充
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [pad_id] * (self.max_len - len(ids))
        
        return torch.tensor([ids], dtype=torch.long)
    
    def predict(self, text: str, threshold: float = 0.2, return_probs: bool = False):
        """预测单条文本"""
        input_ids = self.text_to_ids(text).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # 获取预测标签
        pred_labels = []
        prob_dict = {}
        for idx, prob in enumerate(probs):
            label_name = self.labels.get(idx, f"label_{idx}")
            prob_dict[label_name] = float(prob)
            if prob > threshold:
                pred_labels.append(label_name)
        
        result = {
            'text': text,
            'labels': pred_labels,
        }
        
        if return_probs:
            result['probabilities'] = prob_dict
        
        return result


# 创建 FastAPI 应用
app = FastAPI(title="TextCNN 文本分类服务", version="1.0.0")
service = None


@app.on_event("startup")
async def startup_event():
    global service
    # 从环境变量或默认路径加载模型
    model_dir = os.environ.get('MODEL_DIR', 'checkpoints')
    service = TextCNNService(model_dir)
    print(f"Service started with model from: {model_dir}")


@app.get("/")
async def root():
    return {
        "message": "TextCNN 文本分类服务",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": service is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = service.predict(
            request.text,
            threshold=request.threshold,
            return_probs=request.return_probs
        )
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(requests: List[PredictRequest]):
    """批量预测"""
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for req in requests:
        try:
            result = service.predict(
                req.text,
                threshold=req.threshold,
                return_probs=req.return_probs
            )
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "text": req.text})
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Start TextCNN API server')
    parser.add_argument('--model_dir', type=str, default='checkpoints',
                        help='Directory containing model files')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to bind')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['MODEL_DIR'] = args.model_dir
    
    # 启动服务
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Model directory: {args.model_dir}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

"""使用 BERT + TextCNN 训练"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from src.models.bert_textcnn import BertTextCNN, BertTokenizerWrapper


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, label_map, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['title'] + ' ' + item['text']
        
        # BERT编码
        input_ids, attention_mask = self.tokenizer.encode(text)
        
        # 标签编码
        labels = torch.zeros(len(self.label_map))
        for label in item['labels']:
            if label in self.label_map:
                labels[self.label_map[label]] = 1.0
        
        return input_ids, attention_mask, labels


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for input_ids, attention_mask, labels in tqdm(dataloader, desc="Training"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.2).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    
    return avg_loss, f1, precision, recall


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Validating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.2).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    
    return avg_loss, f1, precision, recall


def main():
    # 加载数据
    with open('data/train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('data/val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # 构建标签映射
    all_labels = set()
    for item in train_data + val_data:
        all_labels.update(item['labels'])
    label_list = sorted(list(all_labels))
    label_map = {label: idx for idx, label in enumerate(label_list)}
    print(f"Labels: {label_list}")
    
    # 初始化tokenizer和模型
    print("Loading BERT model...")
    tokenizer = BertTokenizerWrapper(max_len=512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = BertTextCNN(
        num_classes=len(label_list),
        filter_sizes=[2, 3, 4, 5],
        num_filters=128,
        dropout=0.5,
        freeze_bert=True  # 冻结BERT参数，只训练CNN部分
    ).to(device)
    
    # 数据集
    train_dataset = TextDataset(train_data, tokenizer, label_map)
    val_dataset = TextDataset(val_data, tokenizer, label_map)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # 优化器和损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练
    best_f1 = 0
    patience = 5
    patience_counter = 0
    
    print("\n" + "="*60)
    print("BERT + TextCNN Training")
    print("="*60)
    
    for epoch in range(20):
        print(f"\nEpoch {epoch+1}/20")
        
        train_loss, train_f1, train_p, train_r = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, P: {train_p:.4f}, R: {train_r:.4f}")
        
        val_loss, val_f1, val_p, val_r = eval_epoch(
            model, val_loader, criterion, device
        )
        print(f"Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, P: {val_p:.4f}, R: {val_r:.4f}")
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_map': label_map,
                'best_f1': best_f1
            }, 'checkpoints/bert_best_model.pt')
            print(f"✓ New best model saved! F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best validation F1: {best_f1:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

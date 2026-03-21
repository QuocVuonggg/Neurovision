import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.backends.cudnn as cudnn

from unet_dataset import BrainTumorUNetDataset
from unet_model import UNet
import numpy as np

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 1. BCE Loss giữ ổn định nền tảng (Hoạt động tốt trên cả não bệnh và não khỏe)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # 2. Smart Dice Loss (CHỐNG HỘI CHỨNG MODEL COLLAPSE)
        inputs_sig = torch.sigmoid(inputs)
        batch_size = inputs.shape[0]
        dice_loss = 0.0
        
        # Phân tích từng ảnh một trong lô (Batch)
        for i in range(batch_size):
            inp = inputs_sig[i].view(-1)
            targ = targets[i].view(-1)
            
            if targ.sum() > 0: 
                # TRƯỜNG HỢP 1: BỆNH NHÂN CÓ U 
                # -> Ép Dice Loss để bắt AI phải vẽ kín khối u (không lủng lỗ đen)
                intersection = (inp * targ).sum()
                dice = (2. * intersection + self.smooth) / (inp.sum() + targ.sum() + self.smooth)
                dice_loss += (1.0 - dice)
            else:
                # TRƯỜNG HỢP 2: BỆNH NHÂN NÃO KHỎE (Tr-no)
                # -> Bỏ qua Dice Loss! Tránh việc AI bị phạt oan uổng dẫn đến "sợ không dám đoán"
                pass
                
        dice_loss = dice_loss / batch_size
        
        # Trộn sức mạnh: BCE dập nhiễu nền + Dice bao khuôn khối u
        return bce_loss + dice_loss

def train_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Khởi động U-Net (TURBO MODE) trên: {device}")
    
    # ÉP XUNG GPU: Bật cudnn benchmark để tăng tốc độ các lớp Tích chập (Convolution)
    if torch.cuda.is_available():
        cudnn.benchmark = True

    # BẠN NHỚ GIỮ NGUYÊN ĐƯỜNG DẪN CỦA MÁY BẠN Ở ĐÂY NHÉ:
    DATA_DIR = r"D:\hackathon\Hackathon_2026\Hackathon_2026-main\data\unet_dataset\train" 
    
    dataset = BrainTumorUNetDataset(data_dir=DATA_DIR)
    
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    print(f"Kho dữ liệu: {train_size} ảnh Train | {val_size} ảnh Kiểm tra.")

    # =====================================================================
    # ÉP XUNG RAM TẠI ĐÂY:
    # 1. batch_size = 32 (Nhồi 32 ảnh 1 lúc thay vì 16)
    # 2. num_workers = 8 (Dùng 8 luồng CPU vắt kiệt RAM để chuẩn bị data)
    # 3. pin_memory = True (Khóa chặt RAM, bơm thẳng vào VRAM của GPU)
    # =====================================================================
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = DiceBCELoss()
    
    # Khi tăng batch_size lên gấp đôi (32), theo lý thuyết ta cũng nên tăng learning rate lên 1 chút để model học mạnh hơn
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs('models', exist_ok=True)
    best_loss = float('inf')
    epochs = 30 

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            # non_blocking=True giúp chuyển data từ RAM sang VRAM mượt hơn mà không làm nghẽn CPU
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                preds = model(images)
                loss = criterion(preds, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/unet_best.pth')
            print("   💾 Đã lưu mô hình U-Net cực đỉnh (TURBO MODE)!")

if __name__ == "__main__":
    train_unet()
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class BrainTumorUNetDataset(Dataset):
    def __init__(self, data_dir, image_size=(224, 224)):
        self.data_dir = data_dir
        self.image_size = image_size
        
        # 1. Lọc danh sách ảnh gốc (Bao gồm cả Tr-pi, Tr-gl, Tr-me, Tr-no)
        all_files = os.listdir(data_dir)
        self.image_files = [f for f in all_files if not f.endswith('_m.jpg') and f.endswith('.jpg')]
        
        print(f"✅ Đã nạp thành công {len(self.image_files)} ảnh gốc vào danh sách.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.jpg', '_m.jpg') 
        
        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)
        
        # Đọc ảnh gốc
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"❌ Lỗi đọc ảnh gốc: {img_name}")
            
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        
        # ===============================================================
        # DATA AUGMENTATION: ÉP AI HỌC CÁCH NHÌN VÙNG TỐI (NECROTIC)
        # ===============================================================
        import random
        if random.random() < 0.5:
            # Ngẫu nhiên thay đổi độ tương phản (Contrast) và độ sáng (Brightness)
            alpha = random.uniform(0.7, 1.3) # Tương phản: 0.7 -> 1.3
            beta = random.uniform(-0.15, 0.15) # Độ sáng: -15% -> +15%
            image = np.clip(image * alpha + beta, 0.0, 1.0)
        # ===============================================================

        # Đọc Mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = np.zeros(self.image_size, dtype=np.float32)
        
        # Chuyển sang PyTorch Tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        return image_tensor, mask_tensor

# =====================================================================
# TEST NHANH (Chạy thử để xem code có tự xử lý được Tr-no không)
# =====================================================================
if __name__ == "__main__":
    import random
    
    # BẠN ĐIỀN ĐƯỜNG DẪN THƯ MỤC CHỨA TẤT CẢ ẢNH VÀO ĐÂY
    MY_DATA_DIR = r"D:\hackathon\Hackathon_2026\Hackathon_2026-main\data\unet_dataset\train" 
    
    dataset = BrainTumorUNetDataset(data_dir=MY_DATA_DIR)
    
    if len(dataset) > 0:
        random_idx = random.randint(0, len(dataset) - 1)
        img_tensor, mask_tensor = dataset[random_idx]
        
        img_np = img_tensor.squeeze().numpy()
        mask_np = mask_tensor.squeeze().numpy()
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Ảnh gốc MRI\n{dataset.image_files[random_idx]}")
        plt.imshow(img_np, cmap='gray')
        
        plt.subplot(1, 2, 2)
        # Kiểm tra xem mask có trống không (Dành cho Tr-no)
        if np.max(mask_np) == 0:
            plt.title("Mask Khối U\n(Não khỏe mạnh - Mask đen)")
        else:
            plt.title("Mask Khối U\n(Ground Truth)")
        plt.imshow(mask_np, cmap='gray')
        plt.show()
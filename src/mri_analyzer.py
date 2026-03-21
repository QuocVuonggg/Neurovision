import cv2
import numpy as np
import torch
import torch.nn.functional as F

def get_disease_suggestion(x, y, w, h, img_width=224, img_height=224):
    cx = x + w / 2  
    cy = y + h / 2  
    if cx < img_width * 0.25 or cx > img_width * 0.75 or cy < img_height * 0.25:
        return "Cortical lesion / Meningioma", (0, 0, 255), "Blue" 
    elif img_width * 0.35 < cx < img_width * 0.65 and img_height * 0.4 < cy < img_height * 0.7:
        return "Thalamic / Deep Brain lesion", (255, 0, 0), "Red" 
    else:
        return "White matter lesion / Tumor", (0, 255, 0), "Green"

def analyze_mri_unet(input_tensor, unet_model, original_img):
    orig_uint8 = (original_img * 255).astype(np.uint8)
    h_img, w_img = orig_uint8.shape

    # ==========================================================
    # 1. U-NET DỰ ĐOÁN XÁC SUẤT
    # ==========================================================
    unet_model.eval()
    with torch.no_grad():
        preds = unet_model(input_tensor)
        preds = torch.sigmoid(preds) 
        prob_map = preds.squeeze().cpu().numpy()

    prob_map_resized = cv2.resize(prob_map, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    # ==========================================================
    # 2. RENDER HEATMAP: LAN TỎA CỰC ĐẠI
    # ==========================================================
    # Tăng kích thước bộ tán xạ lên 181x181 để sóng nhiệt bung ra thật xa
    smooth_prob = cv2.GaussianBlur(prob_map_resized, (181, 181), 0)
    
    if smooth_prob.max() > 0:
        smooth_prob = (smooth_prob / smooth_prob.max()) * prob_map_resized.max()

    # Hạ Gamma xuống 0.4: Khuếch đại mãnh liệt các vùng nhiệt độ thấp ở xa tâm
    smooth_prob = np.power(smooth_prob, 0.4)

    _, binary_head = cv2.threshold(orig_uint8, 15, 255, cv2.THRESH_BINARY)
    contours_head, _ = cv2.findContours(binary_head, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brain_mask = np.zeros_like(orig_uint8)
    if contours_head:
        largest_head = max(contours_head, key=cv2.contourArea)
        cv2.drawContours(brain_mask, [largest_head], -1, 255, thickness=cv2.FILLED)

    heatmap_bgr = cv2.applyColorMap((smooth_prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    orig_rgb = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2RGB)
    heatmap_final = orig_rgb.copy()

    # Độ mờ tối đa ở vùng lõi là 0.55 để giữ độ trong suốt dịu mắt
    alpha = np.clip(smooth_prob * 1.2 + 0.20, 0.0, 0.55) 
    
    brain_idx = brain_mask > 0
    for c in range(3):
        heatmap_final[:, :, c][brain_idx] = (
            orig_rgb[:, :, c][brain_idx] * (1.0 - alpha[brain_idx]) +
            heatmap_color[:, :, c][brain_idx] * alpha[brain_idx]
        ).astype(np.uint8)

    # ==========================================================
    # 3. TẠO HỘP BAO QUÁT (SAFETY MARGIN BOUNDING BOX)
    # ==========================================================
    peak_conf = np.max(prob_map_resized)
    
    if peak_conf > 0.12:
        has_anomaly = True
        dynamic_thresh = max(0.03, peak_conf * 0.10)
    else:
        has_anomaly = False
        dynamic_thresh = 1.0 

    binary_mask = (prob_map_resized > dynamic_thresh).astype(np.uint8) * 255
    
    # Bung lưới cực mạnh: Dùng kernel 15x15 và bung liên tiếp 3 vòng (iterations=3)
    # Bước này giúp lưới "ăn" luôn cả khu vực mô mềm lân cận khối u
    kernel_expand = np.ones((2, 2), np.uint8)
    expanded_mask = cv2.dilate(binary_mask, kernel_expand, iterations=3)
    solid_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_CLOSE, kernel_expand)
    
    cnts, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for c in cnts:
        if cv2.contourArea(c) > 60: 
            hull = cv2.convexHull(c) 
            valid_contours.append(hull)

    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

    # ==========================================================
    # 4. VẼ HỘP VÀ XUẤT BÁO CÁO
    # ==========================================================
    img_with_annotations = orig_rgb.copy()
    suggestions_list = []
    prob_abnormal = 0.0

    if has_anomaly and len(valid_contours) > 0:
        prob_abnormal = float(peak_conf) * 100
        
        box_count = 0
        for cnt in valid_contours[:3]: 
            box_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            
            # CỘNG THÊM BIÊN AN TOÀN: Nới hộp ra thêm 20 pixel mỗi chiều
            pad = 20
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad); y2 = min(h_img, y + h + pad)
            
            disease_name, box_color, color_name = get_disease_suggestion(
                x1, y1, x2 - x1, y2 - y1, w_img, h_img
            )

            cv2.rectangle(img_with_annotations, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                img_with_annotations, f"#{box_count}",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, box_color, 1, cv2.LINE_AA,
            )

            local_conf = np.max(prob_map_resized[y1:y2, x1:x2]) * 100
            
            box_pixels = orig_uint8[y1:y2, x1:x2]
            box_mean = np.mean(box_pixels) if len(box_pixels) > 0 else 100

            if box_mean < 75:
                subtype = "Necrotic/Cystic Hole"
            elif box_mean > 135:
                subtype = "Solid Enhancing Spot"
            else:
                subtype = "Tumor Core"

            coord_str = f"X:{x1}➞{x2}, Y:{y1}➞{y2}"
            base_disease = disease_name.split("/")[0].strip()

            suggestions_list.append(
                f"• [#{box_count} - {color_name} Box] {subtype} ({base_disease}) "
                f"| AI Peak Focus: {local_conf:.1f}% | Pos: [{coord_str}]"
            )
    else:
        has_anomaly = False 

    return img_with_annotations, heatmap_final, prob_abnormal, has_anomaly, suggestions_list
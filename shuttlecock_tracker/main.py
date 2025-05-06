import os
import cv2
import numpy as np
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
import argparse
from shuttlecock_detector import detect_shuttlecock_sequence

def get_available_device(requested_device="cuda"):
    """
    检查并返回可用的设备
    
    Args:
        requested_device (str): 请求使用的设备
    
    Returns:
        str: 可用的设备名称
    """
    if requested_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif requested_device == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def detect_shuttlecocks(video_path, device="cuda"):
    """
    使用SAM2模型检测和追踪羽毛球视频中的羽毛球。
    
    Args:
        video_path (str): 输入羽毛球视频的路径
        device (str): 使用的设备，默认为"cuda"
    """
    # 检查可用设备
    actual_device = get_available_device(device)
    if actual_device != device:
        print(f"警告：请求的设备 '{device}' 不可用，切换到 '{actual_device}'")
    
    device = actual_device
    print(f"使用设备: {device}")
    
    print(f"正在加载SAM2模型...")
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large", device=device)
    predictor.to(device=device)
    
    print(f"正在处理视频: {video_path}")
    
    # 读取输入视频以获取帧率和尺寸等信息
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"视频帧率: {fps}, 宽度: {width}, 高度: {height}, 帧数: {frame_count}")

    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):        
        # 自动检测羽毛球位置
        shuttlecock_pos = detect_shuttlecock_sequence(video_path)
        
        if shuttlecock_pos:
            center_x, center_y, radius = shuttlecock_pos
            print(f"检测到羽毛球位置: 中心=({center_x}, {center_y}), 半径={radius}")
        else:
            # 使用默认位置
            center_x, center_y = width // 2, height // 2
            radius = 30
            print(f"未检测到羽毛球，使用默认位置: 中心=({center_x}, {center_y}), 半径={radius}")
        
        # 创建边界框 [x1, y1, x2, y2]
        # 使用稍大的半径以确保羽毛球完全在框内
        box_radius = int(radius * 1.5)
        box = [
            max(0, center_x - box_radius), 
            max(0, center_y - box_radius), 
            min(width, center_x + box_radius), 
            min(height, center_y + box_radius)
        ]

        # 初始化SAM2状态
        print(f"初始化SAM2...")
        state = predictor.init_state(video_path)
        print(f"初始化SAM2完成")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return
        ret, frame = cap.read()
        if not ret:
            print(f"错误：无法读取视频文件 {video_path}")
            return

        frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, 0, 0, box=box)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"错误：无法读取视频文件 {video_path}")
                return
            
            mask = masks[0][0]
            
            # 将mask从PyTorch张量转换为NumPy数组
            mask_np = mask.cpu().numpy()
            
            # 创建彩色遮罩以突出显示目标区域
            # 将掩码转换为uint8类型，值为0-255
            mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
            
            # 创建彩色遮罩 (红色)
            colored_mask = np.zeros_like(frame)
            colored_mask[:,:,2] = mask_binary  # 将红色通道设置为掩码值
            
            # 将掩码与原始帧融合
            alpha = 0.5  # 透明度
            cv2.addWeighted(colored_mask, alpha, frame, 1-alpha, 0, frame)
            
            # 在原始帧上显示轮廓
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)  # 绿色轮廓
            cv2.imshow("Shuttlecock Detection", frame)
            cv2.waitKey(30)

        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="羽毛球检测和追踪工具")
    parser.add_argument("--input", type=str, required=True, help="输入视频路径")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="使用的设备 (cuda/mps/cpu)，将自动检测可用性")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误：输入视频 {args.input} 不存在")
        return

    detect_shuttlecocks(args.input, args.device)

if __name__ == "__main__":
    main() 
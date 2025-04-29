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

def detect_shuttlecocks(video_path, output_path, device="cuda"):
    """
    使用SAM2模型检测和追踪羽毛球视频中的羽毛球。
    
    Args:
        video_path (str): 输入羽毛球视频的路径
        output_path (str): 输出追踪结果视频的路径
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
    
    # 确定输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"正在处理视频: {video_path}")
    
    # 读取输入视频以获取帧率和尺寸等信息
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # 设置输出视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float32 if device == "cpu" else torch.bfloat16):
        # 初始化SAM2状态
        state = predictor.init_state(video_path)
        
        # 获取第一帧并初始化追踪
        first_frame_idx = 0
        
        # 自动检测羽毛球位置
        print("尝试自动检测羽毛球位置...")
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
        
        # 添加初始边界框作为提示
        object_id = 1  # 对象ID
        _, object_ids, masks = predictor.add_new_points_or_box(
            state, 
            frame_idx=first_frame_idx, 
            obj_id=object_id,
            box=box
        )
        
        # 显示初始化结果
        frame_with_mask = visualize_mask(state["images"][first_frame_idx].cpu().numpy(), masks.cpu().numpy())
        
        # 添加羽毛球位置标记
        cv2.circle(frame_with_mask, (center_x, center_y), radius, (0, 255, 0), 2)
        cv2.rectangle(
            frame_with_mask, 
            (box[0], box[1]), 
            (box[2], box[3]), 
            (255, 255, 0), 
            2
        )
        
        out.write(cv2.cvtColor(frame_with_mask, cv2.COLOR_RGB2BGR))
        print(f"初始化追踪成功，开始追踪视频中的羽毛球...")
        
        # 在整个视频中传播掩码（追踪羽毛球）
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            # 转换为NumPy数组并可视化
            frame = state["images"][frame_idx].cpu().numpy()
            frame_with_mask = visualize_mask(frame, masks.cpu().numpy())
            
            # 保存帧
            out.write(cv2.cvtColor(frame_with_mask, cv2.COLOR_RGB2BGR))
            
            # 显示进度
            print(f"处理帧: {frame_idx+1}/{frame_count}", end="\r")
    
    out.release()
    print(f"\n视频处理完成，输出保存到: {output_path}")

def visualize_mask(frame, masks):
    """
    在帧上可视化掩码
    
    Args:
        frame: 视频帧
        masks: 掩码数据
    
    Returns:
        带有掩码的帧
    """
    # 确保frame是RGB格式并且值在0-255范围内
    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
    
    # 创建一个彩色掩码覆盖层
    color_mask = np.zeros_like(frame)
    
    # 使用明亮的颜色表示羽毛球
    color = [255, 0, 0]  # 红色
    
    # 将掩码应用到颜色层
    for i in range(3):
        color_mask[:, :, i] = color[i] * masks[0]
    
    # 将掩码与原始帧混合
    alpha = 0.5  # 透明度
    frame_with_mask = cv2.addWeighted(frame, 1, color_mask, alpha, 0)
    
    # 在检测到的羽毛球位置绘制轮廓
    contours, _ = cv2.findContours(
        (masks[0] * 255).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(frame_with_mask, contours, -1, (0, 255, 0), 2)  # 绿色轮廓
    
    return frame_with_mask

def main():
    parser = argparse.ArgumentParser(description="羽毛球检测和追踪工具")
    parser.add_argument("--input", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output", type=str, default="output.mp4", help="输出视频路径")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="使用的设备 (cuda/mps/cpu)，将自动检测可用性")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误：输入视频 {args.input} 不存在")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    detect_shuttlecocks(args.input, args.output, args.device)

if __name__ == "__main__":
    main() 
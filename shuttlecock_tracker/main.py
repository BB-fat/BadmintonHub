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

def detect_shuttlecocks(video_path, device="cuda", window_size=60, overlap=10):
    """
    使用SAM2模型检测和追踪羽毛球视频中的羽毛球。
    使用滑动窗口技术处理视频，避免MPS后端的NDArray大小限制。
    
    Args:
        video_path (str): 输入羽毛球视频的路径
        device (str): 使用的设备，默认为"cuda"
        window_size (int): 滑动窗口的帧数，默认为60帧
        overlap (int): 窗口之间的重叠帧数，默认为10帧
    """
    # 检查可用设备
    actual_device = get_available_device(device)
    if actual_device != device:
        print(f"警告：请求的设备 '{device}' 不可用，切换到 '{actual_device}'")
    
    device = actual_device
    print(f"使用设备: {device}")
    
    print(f"正在加载SAM2模型...")
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-small", device=device)
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
    
    # 使用滑动窗口处理视频
    start_frame = 0
    last_box = None  # 存储上一个窗口最后一帧的边界框
    
    while start_frame < frame_count:
        # 计算当前窗口的结束帧
        end_frame = min(start_frame + window_size, frame_count)
        
        print(f"处理帧 {start_frame} 到 {end_frame-1} (共 {frame_count} 帧)")
        
        # 清理GPU内存
        if device == "cuda" or device == "mps":
            torch.cuda.empty_cache() if device == "cuda" else torch.mps.empty_cache()
        
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            # 创建临时视频片段
            temp_video_path = f"temp_segment_{start_frame}_{end_frame}.mp4"
            extract_video_segment(video_path, temp_video_path, start_frame, end_frame, fps)
            
            try:
                # 初始化SAM2状态
                print(f"初始化SAM2...")
                state = predictor.init_state(temp_video_path)
                print(f"初始化SAM2完成")
                
                # 打开临时视频
                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    print(f"错误：无法打开视频文件 {temp_video_path}")
                    start_frame = end_frame - overlap
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    print(f"错误：无法读取视频文件 {temp_video_path}")
                    start_frame = end_frame - overlap
                    continue
                
                # 添加初始框或使用上一个窗口的边界框
                if last_box is not None and start_frame > 0:
                    print(f"使用上一个窗口的边界框继续追踪: {last_box}")
                    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, 0, 0, box=last_box)
                else:
                    print(f"使用初始box开始追踪: {box}")
                    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, 0, 0, box=box)
                
                # 用于存储最后一帧的边界框
                final_box = None
                
                # 处理所有帧
                for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                    # 获取当前帧
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 处理掩码
                    if len(masks) > 0 and len(masks[0]) > 0:
                        mask = masks[0][0]
                        
                        # 将mask从PyTorch张量转换为NumPy数组
                        mask_np = mask.cpu().numpy()
                        
                        # 创建二值掩码
                        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                        
                        # 如果是最后一帧，计算边界框用于下一个窗口
                        if frame_idx == min(window_size - 1, end_frame - start_frame - 1):
                            # 查找非零点的坐标
                            non_zero_points = np.where(mask_binary > 0)
                            if len(non_zero_points[0]) > 0:  # 确保有非零点
                                # 计算边界框坐标 [x1, y1, x2, y2]
                                y_min, y_max = non_zero_points[0].min(), non_zero_points[0].max()
                                x_min, x_max = non_zero_points[1].min(), non_zero_points[1].max()
                                
                                # 添加一些边距
                                padding = 5
                                x_min = max(0, x_min - padding)
                                y_min = max(0, y_min - padding)
                                x_max = min(width, x_max + padding)
                                y_max = min(height, y_max + padding)
                                
                                final_box = [int(x_min), int(y_min), int(x_max), int(y_max)]
                                print(f"最后一帧的边界框: {final_box}")
                        
                        # 创建彩色遮罩 (红色)
                        colored_mask = np.zeros_like(frame)
                        colored_mask[:,:,2] = mask_binary  # 将红色通道设置为掩码值
                        
                        # 将掩码与原始帧融合
                        alpha = 0.5  # 透明度
                        cv2.addWeighted(colored_mask, alpha, frame, 1-alpha, 0, frame)
                        
                        # 在原始帧上显示轮廓
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)  # 绿色轮廓
                    
                    # 显示当前处理的帧号
                    global_frame_idx = start_frame + frame_idx
                    cv2.putText(frame, f"Frame: {global_frame_idx}/{frame_count}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    cv2.imshow("Shuttlecock Detection", frame)
                    key = cv2.waitKey(30)
                    if key == 27:  # ESC键 - 停止处理
                        break
                
                # 更新last_box为当前窗口的最后一帧边界框
                if final_box is not None:
                    last_box = final_box
                
                cap.release()
                
                # 如果用户按ESC键停止处理，则退出整个循环
                if key == 27:
                    break
                    
            finally:
                # 清理临时文件
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        
        # 移动到下一个窗口，保留一些重叠以确保平滑过渡
        start_frame = end_frame - overlap
    
    cv2.destroyAllWindows()

def extract_video_segment(input_video, output_video, start_frame, end_frame, fps):
    """
    从输入视频中提取特定范围的帧并保存为新的视频文件
    
    Args:
        input_video (str): 输入视频路径
        output_video (str): 输出视频路径
        start_frame (int): 起始帧号
        end_frame (int): 结束帧号
        fps (float): 帧率
    """
    cap = cv2.VideoCapture(input_video)
    
    # 设置起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 获取视频尺寸
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 提取并写入帧
    frame_count = start_frame
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
    
    # 释放资源
    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description="羽毛球检测和追踪工具")
    parser.add_argument("--input", type=str, required=True, help="输入视频路径")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="使用的设备 (cuda/mps/cpu)，将自动检测可用性")
    parser.add_argument("--window-size", type=int, default=60,
                        help="滑动窗口的帧数，默认为60")
    parser.add_argument("--overlap", type=int, default=10,
                        help="窗口之间的重叠帧数，默认为10")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误：输入视频 {args.input} 不存在")
        return

    detect_shuttlecocks(args.input, args.device, args.window_size, args.overlap)

if __name__ == "__main__":
    main() 
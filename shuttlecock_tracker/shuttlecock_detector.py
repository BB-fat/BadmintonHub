import cv2
import numpy as np

def detect_shuttlecock_in_frame(frame):
    """
    尝试自动检测帧中的羽毛球。
    
    这是一个基于图像处理的简单方法，通过以下步骤检测羽毛球：
    1. 将帧转换为灰度图
    2. 使用高斯模糊减少噪声
    3. 使用Canny边缘检测器检测边缘
    4. 查找可能是羽毛球的圆形
    
    Args:
        frame: 视频帧 (numpy数组)
        
    Returns:
        tuple: (center_x, center_y, radius) 如果检测到羽毛球，否则返回None
    """
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 羽毛球的可能特征
    shuttlecock_candidates = []
    
    # 遍历所有轮廓
    for contour in contours:
        # 计算轮廓的面积和周长
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 忽略太小或太大的轮廓
        if area < 20 or area > 500:
            continue
        
        # 圆度 = 4π * 面积 / 周长²
        # 圆的圆度接近1
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 羽毛球应该近似圆形
            if 0.5 < circularity < 1.0:
                # 轮廓的最小包围圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # 计算亮度（羽毛球通常比背景亮）
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_brightness = cv2.mean(gray, mask=mask)[0]
                
                shuttlecock_candidates.append({
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'circularity': circularity,
                    'area': area,
                    'brightness': mean_brightness
                })
    
    # 如果没有候选项，返回None
    if not shuttlecock_candidates:
        return None
    
    # 根据羽毛球特征（圆度、亮度、大小）对候选项排序
    # 这里我们认为羽毛球应该是一个较亮、较圆的小物体
    shuttlecock_candidates.sort(key=lambda x: (
        -x['circularity'],  # 越圆越好
        -x['brightness'],   # 越亮越好
        x['area']           # 合适的大小
    ))
    
    # 取最有可能的候选项
    best_candidate = shuttlecock_candidates[0]
    
    return (
        best_candidate['center'][0], 
        best_candidate['center'][1], 
        best_candidate['radius']
    )

def detect_with_motion(prev_frame, curr_frame, next_frame=None):
    """
    使用运动检测来找到羽毛球。
    这种方法在相机固定的情况下效果较好。
    
    Args:
        prev_frame: 前一帧
        curr_frame: 当前帧
        next_frame: 下一帧（可选）
        
    Returns:
        tuple: (center_x, center_y, radius) 如果检测到羽毛球，否则返回None
    """
    # 确保所有帧为灰度图
    if len(prev_frame.shape) > 2:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    else:
        prev_gray = prev_frame
        
    if len(curr_frame.shape) > 2:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    else:
        curr_gray = curr_frame
    
    # 计算帧差
    frame_diff = cv2.absdiff(prev_gray, curr_gray)
    
    # 如果有下一帧，也计算与下一帧的差
    if next_frame is not None:
        if len(next_frame.shape) > 2:
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        else:
            next_gray = next_frame
        
        next_diff = cv2.absdiff(curr_gray, next_gray)
        # 结合两个差异
        frame_diff = cv2.bitwise_and(frame_diff, next_diff)
    
    # 应用阈值
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    
    # 应用形态学操作来去除噪声
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 羽毛球候选
    shuttlecock_candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 羽毛球通常是小物体
        if 20 < area < 500:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            shuttlecock_candidates.append({
                'center': (int(x), int(y)),
                'radius': int(radius),
                'area': area
            })
    
    if not shuttlecock_candidates:
        return None
    
    # 选择面积适中的候选项
    shuttlecock_candidates.sort(key=lambda x: abs(x['area'] - 200))
    best_candidate = shuttlecock_candidates[0]
    
    return (
        best_candidate['center'][0], 
        best_candidate['center'][1], 
        best_candidate['radius']
    )

def detect_shuttlecock_sequence(video_path, num_frames=10):
    """
    从视频的前几帧中检测羽毛球。
    
    Args:
        video_path: 视频文件路径
        num_frames: 要检查的帧数
    
    Returns:
        tuple: (center_x, center_y, radius) 如果检测到羽毛球，否则返回None
    """
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None
    
    # 读取前num_frames帧
    frames = []
    for _ in range(min(num_frames, 30)):  # 限制最多30帧以防视频太长
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    
    if len(frames) < 2:
        print("视频帧数不足，无法进行运动检测")
        return None
    
    # 尝试单帧检测
    for i, frame in enumerate(frames):
        result = detect_shuttlecock_in_frame(frame)
        if result:
            print(f"在第{i+1}帧使用单帧检测方法找到羽毛球")
            return result
    
    # 尝试运动检测
    for i in range(len(frames) - 1):
        prev_frame = frames[i]
        curr_frame = frames[i + 1]
        next_frame = frames[i + 2] if i + 2 < len(frames) else None
        
        result = detect_with_motion(prev_frame, curr_frame, next_frame)
        if result:
            print(f"在第{i+1}-{i+2}帧使用运动检测方法找到羽毛球")
            return result
    
    print("无法自动检测到羽毛球，使用视频中心点作为默认位置")
    # 如果所有检测都失败，使用视频中心和默认半径
    if frames:
        height, width = frames[0].shape[:2]
        return width // 2, height // 2, 30
    
    return None 
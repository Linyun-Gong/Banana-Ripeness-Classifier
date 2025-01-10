import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops   
from ultralytics import YOLO

# 加载 YOLO 模型
model = None

def preprocess_image(image_input, target_size=(960, 540)):
    """图像预处理函数
    
    Args:
        image_input: 输入图像(可以是路径字符串或numpy数组)
        target_size: 目标图像尺寸，默认 (960, 540)
        
    Returns:
        tuple: (对比度增强后的图像, HSV图像)
    """
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError("无法读取图像文件")
    else:
        if not isinstance(image_input, np.ndarray):
            raise ValueError("输入必须是图像路径或numpy数组")
        if len(image_input.shape) != 3:
            raise ValueError("输入图像必须是3通道彩色图像")
        image = image_input.copy()
    
    # 检查图像是否为空
    if image.size == 0:
        raise ValueError("输入图像为空")
    
    # 确保图像类型正确
    image = image.astype(np.uint8)
    
    # 检查图像方向并旋转
    h, w = image.shape[:2]
    if h > w:  # 如果是竖着的图片
        print(f"Rotating image from {w}x{h} to {h}x{w}")
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    # 统一图像尺寸
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
    # 双边滤波去噪但保留边缘
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 光照归一化
    try:
        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # 确保 L 通道是 uint8 类型
        l = l.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"CLAHE error: {str(e)}")
        enhanced = filtered  # 如果处理失败，使用原始滤波图像
    
    # 自适应伽马校正
    def adjust_gamma(image):
        mean = np.mean(image)
        gamma = np.log(0.5)/np.log(mean/255)
        gamma_corrected = np.power(image/255.0, gamma)
        return (gamma_corrected * 255).astype(np.uint8)
    
    enhanced = adjust_gamma(enhanced)
    
    # 对比度增强
    alpha = 1.3
    contrast = cv2.convertScaleAbs(enhanced, alpha=alpha)

    # 转换为HSV空间
    hsv_image = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    
    return contrast, hsv_image

def postprocess_mask(mask, min_area=1000):
    """对分割掩码进行后处理
    
    Args:
        mask: 二值掩码图像
        min_area: 最小区域面积阈值
    
    Returns:
        处理后的掩码
    """
    # 形态学操作清理噪声
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 轮廓处理
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_mask = np.zeros_like(mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # 轮廓平滑
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(refined_mask, [approx], -1, (255), -1)
    
    # 边缘平滑
    refined_mask = cv2.GaussianBlur(refined_mask, (5,5), 0)
    refined_mask = (refined_mask > 127).astype(np.uint8) * 255
    
    return refined_mask

def tamura_coarseness(gray_image, mask, kmax=4):
    """计算香蕉区域的Tamura粗糙度特征
    
    Args:
        gray_image: 灰度图像
        mask: 香蕉区域掩码
        kmax: 最大窗口尺寸的指数
        
    Returns:
        float: 粗糙度值
    """
    # 只保留香蕉区域的灰度值
    masked_gray = cv2.bitwise_and(gray_image, mask)
    
    h, w = masked_gray.shape
    kmax = min(kmax, int(np.floor(np.log2(min(h, w)))))
    average_matrices = []
    
    # 计算不同尺度的平均值
    for k in range(kmax):
        window_size = 2**k
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        average_matrix = cv2.filter2D(masked_gray.astype(float), -1, kernel)
        average_matrices.append(average_matrix)
    
    # 计算差值
    fcrs = np.zeros((h, w))
    for k in range(kmax):
        size = 2**k
        horizontal = np.abs(
            np.roll(average_matrices[k], size, axis=1) - 
            np.roll(average_matrices[k], -size, axis=1)
        )
        vertical = np.abs(
            np.roll(average_matrices[k], size, axis=0) - 
            np.roll(average_matrices[k], -size, axis=0)
        )
        
        current_fcrs = np.maximum(horizontal, vertical) * (2**k)
        fcrs = np.maximum(fcrs, current_fcrs)
    
    # 只计算香蕉区域内的平均值
    masked_fcrs = cv2.bitwise_and(fcrs.astype(np.uint8), mask)
    total_pixels = np.sum(mask > 0)
    return np.sum(masked_fcrs) / total_pixels if total_pixels > 0 else 0

def tamura_contrast(gray_image, mask):
    """计算香蕉区域的Tamura对比度特征
    
    Args:
        gray_image: 灰度图像
        mask: 香蕉区域掩码
        
    Returns:
        float: 对比度值
    """
    # 只考虑香蕉区域的像素
    masked_gray = cv2.bitwise_and(gray_image, mask)
    valid_pixels = masked_gray[mask > 0]
    
    if len(valid_pixels) == 0:
        return 0.0
    
    mean = np.mean(valid_pixels)
    variance = np.var(valid_pixels)
    std = np.sqrt(variance)
    
    if std < 1e-6:
        return 0.0
    
    diff = valid_pixels - mean
    kurtosis = np.mean(diff ** 4) / (variance ** 2 + 1e-8) - 3
    kurtosis_adjusted = max(kurtosis, 0)
    
    alpha4 = kurtosis_adjusted / (std ** 4 + 1e-8)
    denominator = np.power(alpha4 + 1e-8, 0.25)
    contrast = std / denominator if denominator > 0 else 0.0
    
    return float(contrast)

def tamura_directionality(gray_image, mask, num_bins=16):
    """计算香蕉区域的Tamura方向性特征
    
    Args:
        gray_image: 灰度图像
        mask: 香蕉区域掩码
        num_bins: 方向直方图的bin数量
        
    Returns:
        float: 方向性值
    """
    # 只在香蕉区域计算梯度
    masked_gray = cv2.bitwise_and(gray_image, mask)
    
    gx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) + np.pi
    
    # 只考虑香蕉区域内的强梯度点
    threshold = np.mean(magnitude[mask > 0])
    valid_mask = (magnitude > threshold) & (mask > 0)
    
    if not np.any(valid_mask):
        return 0.0
    
    hist, _ = np.histogram(direction[valid_mask], bins=num_bins, range=(0, 2*np.pi))
    hist = hist / np.sum(hist)
    
    directionality = np.var(hist)
    
    return float(directionality)

def segment_banana(image, min_area=1000):
    """使用Otsu阈值法分割香蕉区域
    
    Args:
        image: 输入的BGR图像
        min_area: 最小区域面积阈值，用于过滤噪声
        
    Returns:
        tuple: (原始图像, 分割掩码, 分割结果)
    """
    # 转换到LAB色彩空间，提取a通道（绿色到红色）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 对a通道进行高斯模糊以减少噪声
    a_blurred = cv2.GaussianBlur(a, (5, 5), 0)
    
    # 应用Otsu阈值法
    _, mask = cv2.threshold(a_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形态学操作改善分割结果
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 寻找并筛选轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_mask = np.zeros_like(mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # 填充轮廓
        cv2.drawContours(refined_mask, [contour], -1, (255), -1)
    
    # 应用高斯模糊平滑边缘
    refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
    refined_mask = (refined_mask > 127).astype(np.uint8) * 255
    
    # 生成分割结果
    result = cv2.bitwise_and(image, image, mask=refined_mask)
    
    return image, refined_mask, result

def ripeness_factor(hsv_image, mask):
    """计算成熟度系数
    
    Args:
        hsv_image: HSV色彩空间的图像
        mask: 香蕉区域掩码
        
    Returns:
        float: 成熟度系数(0~1)
    """
    gray = cv2.cvtColor(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    banana_gray = cv2.bitwise_and(gray, gray, mask=mask)
    banana_gray = cv2.GaussianBlur(banana_gray, (5, 5), 0)
    
    _, dark_spots = cv2.threshold(banana_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    dark_spots = cv2.morphologyEx(dark_spots, cv2.MORPH_OPEN, kernel)
    dark_spots = cv2.morphologyEx(dark_spots, cv2.MORPH_CLOSE, kernel)
    dark_spots = cv2.bitwise_and(dark_spots, mask)
    
    total_area = np.sum(mask > 0)
    dark_area = np.sum(dark_spots > 0)
    
    ripeness_factor = min(dark_area / total_area if total_area > 0 else 0, 1.0)
    
    return ripeness_factor

def tamura_coarseness(gray_image, mask, kmax=4):
    """计算香蕉区域的Tamura粗糙度特征
    
    Args:
        gray_image: 灰度图像
        mask: 香蕉区域掩码
        kmax: 最大窗口尺寸的指数
        
    Returns:
        float: 粗糙度值
    """
    # 只保留香蕉区域的灰度值
    masked_gray = cv2.bitwise_and(gray_image, mask)
    
    h, w = masked_gray.shape
    kmax = min(kmax, int(np.floor(np.log2(min(h, w)))))
    average_matrices = []
    
    # 计算不同尺度的平均值
    for k in range(kmax):
        window_size = 2**k
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        average_matrix = cv2.filter2D(masked_gray.astype(float), -1, kernel)
        average_matrices.append(average_matrix)
    
    # 计算差值
    fcrs = np.zeros((h, w))
    for k in range(kmax):
        size = 2**k
        horizontal = np.abs(
            np.roll(average_matrices[k], size, axis=1) - 
            np.roll(average_matrices[k], -size, axis=1)
        )
        vertical = np.abs(
            np.roll(average_matrices[k], size, axis=0) - 
            np.roll(average_matrices[k], -size, axis=0)
        )
        
        current_fcrs = np.maximum(horizontal, vertical) * (2**k)
        fcrs = np.maximum(fcrs, current_fcrs)
    
    # 只计算香蕉区域内的平均值
    masked_fcrs = cv2.bitwise_and(fcrs.astype(np.uint8), mask)
    total_pixels = np.sum(mask > 0)
    return np.sum(masked_fcrs) / total_pixels if total_pixels > 0 else 0

def tamura_contrast(gray_image, mask):
    """计算香蕉区域的Tamura对比度特征
    
    Args:
        gray_image: 灰度图像
        mask: 香蕉区域掩码
        
    Returns:
        float: 对比度值
    """
    # 只考虑香蕉区域的像素
    masked_gray = cv2.bitwise_and(gray_image, mask)
    valid_pixels = masked_gray[mask > 0]
    
    if len(valid_pixels) == 0:
        return 0.0
    
    mean = np.mean(valid_pixels)
    variance = np.var(valid_pixels)
    std = np.sqrt(variance)
    
    if std < 1e-6:
        return 0.0
    
    diff = valid_pixels - mean
    kurtosis = np.mean(diff ** 4) / (variance ** 2 + 1e-8) - 3
    kurtosis_adjusted = max(kurtosis, 0)
    
    alpha4 = kurtosis_adjusted / (std ** 4 + 1e-8)
    denominator = np.power(alpha4 + 1e-8, 0.25)
    contrast = std / denominator if denominator > 0 else 0.0
    
    return float(contrast)

def tamura_directionality(gray_image, mask, num_bins=16):
    """计算香蕉区域的Tamura方向性特征
    
    Args:
        gray_image: 灰度图像
        mask: 香蕉区域掩码
        num_bins: 方向直方图的bin数量
        
    Returns:
        float: 方向性值
    """
    # 只在香蕉区域计算梯度
    masked_gray = cv2.bitwise_and(gray_image, mask)
    
    gx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) + np.pi
    
    # 只考虑香蕉区域内的强梯度点
    threshold = np.mean(magnitude[mask > 0])
    valid_mask = (magnitude > threshold) & (mask > 0)
    
    if not np.any(valid_mask):
        return 0.0
    
    hist, _ = np.histogram(direction[valid_mask], bins=num_bins, range=(0, 2*np.pi))
    hist = hist / np.sum(hist)
    
    directionality = np.var(hist)
    
    return float(directionality)

def extract_features(image_input):
    """提取香蕉图像的特征向量
    
    Args:
        image_input: 输入图像(可以是路径字符串或numpy数组)
        
    Returns:
        ndarray: 4维特征向量 [Tamura对比度, Tamura方向性, HSV色相均值, 成熟度系数]
    """
    try:
        # 预处理图像
        contrast_img, hsv_image = preprocess_image(image_input)
        
        # 提取掩码
        _, banana_mask, _ = segment_banana(contrast_img)
        
        # 转为灰度图
        gray = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)
        
        # 提取特征 - 现在都使用掩码
        tamura_cont = tamura_contrast(gray, banana_mask)
        tamura_dir = tamura_directionality(gray, banana_mask)
        
        # 只计算香蕉区域的HSV色相均值
        masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=banana_mask)
        valid_hue = masked_hsv[:,:,0][banana_mask > 0]
        hue_mean = np.mean(valid_hue) if len(valid_hue) > 0 else 0
        
        # 计算成熟度系数
        rf = ripeness_factor(hsv_image, banana_mask)
        
        # 组合特征向量
        feature_vector = np.array([
            tamura_cont,  # Tamura对比度
            tamura_dir,   # Tamura方向性
            hue_mean,     # HSV色相均值
            rf           # 成熟度系数
        ])
        
        # 处理异常值
        feature_vector = np.nan_to_num(feature_vector, 0)
        
        return feature_vector
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return np.zeros(4)  # 返回零向量作为默认值
import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from ultralytics import YOLO

model = YOLO('yolov8x-seg.pt')

def standardize_image_size(image, target_size=(224, 224)):
    height, width = image.shape[:2]
   
    if height > width:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
        height, width = width, height
  
    scale = min(target_size[0] / width, target_size[1] / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas

def preprocess_image(image):
    standardized = standardize_image_size(image)
  
    filtered = cv2.bilateralFilter(standardized, d=9, sigmaColor=75, sigmaSpace=75)
    
    original_image, mask, _ = segment_banana(filtered)
    
    binary_mask = mask > 0
    
    banana_region = cv2.bitwise_and(filtered, filtered, mask=mask)
    
    lab = cv2.cvtColor(banana_region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
  
    def adjust_gamma(image, mask):
        valid_pixels = image[mask > 0]
        if len(valid_pixels) > 0:
            mean = np.mean(valid_pixels)
            gamma = np.log(0.5)/np.log(mean/255)
            gamma_corrected = np.power(image/255.0, gamma)
            return (gamma_corrected * 255).astype(np.uint8)
        return image
    
    enhanced = adjust_gamma(enhanced, mask)
    
    alpha = 1.3
    contrast = cv2.convertScaleAbs(enhanced, alpha=alpha)
   
    hsv_image = cv2.cvtColor(banana_region, cv2.COLOR_BGR2HSV)
    
    return contrast, hsv_image, mask

def color_features(hsv_image, mask):
    h, s, v = cv2.split(hsv_image)
    feature_list = []
    
    for channel in [h, s, v]:
        masked_channel = cv2.bitwise_and(channel, channel, mask=mask)
        valid_pixels = masked_channel[mask > 0]
        if len(valid_pixels) > 0:
            feature_list.extend([
                np.mean(valid_pixels),
                np.std(valid_pixels),
                np.median(valid_pixels)
            ])
        else:
            feature_list.extend([0, 0, 0])  
    
    return feature_list

def tamura_coarseness(gray_image, kmax=4):
    h, w = gray_image.shape
    kmax = min(kmax, int(np.floor(np.log2(min(h, w)))))
    average_matrices = []
    for k in range(kmax):
        window_size = 2**k
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        average_matrix = cv2.filter2D(gray_image.astype(float), -1, kernel)
        average_matrices.append(average_matrix)
    
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
    
    return np.mean(fcrs)

def tamura_contrast(gray_image):
    mean = np.mean(gray_image)
    variance = np.var(gray_image)
    std = np.sqrt(variance)
   
    if std < 1e-6:
        return 0.0
    
    diff = gray_image - mean
    
    kurtosis = np.mean(diff ** 4) / (variance ** 2 + 1e-8) - 3
    
    kurtosis_adjusted = max(kurtosis, 0) 
    alpha4 = kurtosis_adjusted / (std ** 4 + 1e-8)
    
    denominator = np.power(alpha4 + 1e-8, 0.25)
    contrast = std / denominator if denominator > 0 else 0.0
    
    return float(contrast)

def tamura_directionality(gray_image, num_bins=16):
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) + np.pi 
    
    threshold = np.mean(magnitude)
    mask = magnitude > threshold
    
    if not np.any(mask):
        return 0.0

    hist, _ = np.histogram(direction[mask], bins=num_bins, range=(0, 2*np.pi))
    hist = hist / np.sum(hist)  
    
    directionality = np.var(hist)  
    
    return float(directionality)

def texture_features(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    kernel = np.ones((3,3), np.uint8)
    enhanced = cv2.morphologyEx(masked_gray, cv2.MORPH_CLOSE, kernel)
    
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    valid_pixels = enhanced[mask > 0]
    if len(valid_pixels) > 0:
        valid_pixels_2d = valid_pixels.reshape(-1, 1)
        glcm = graycomatrix(valid_pixels_2d, distances, angles, 256, 
                           symmetric=True, normed=True)
        
        features = {
            'glcm_contrast': graycoprops(glcm, 'contrast')[0,0],
            'glcm_homogeneity': graycoprops(glcm, 'homogeneity')[0,0],
            'glcm_energy': graycoprops(glcm, 'energy')[0,0],
            'glcm_entropy': graycoprops(glcm, 'entropy')[0,0]
        }
    else:
        features = {
            'glcm_contrast': 0,
            'glcm_homogeneity': 0,
            'glcm_energy': 0,
            'glcm_entropy': 0
        }
 
    features.update({
        'tamura_coarseness': tamura_coarseness(enhanced),
        'tamura_contrast': tamura_contrast(enhanced),
        'tamura_directionality': tamura_directionality(enhanced)
    })
    
    return features


def postprocess_mask(mask, min_area=1000):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_mask = np.zeros_like(mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        cv2.drawContours(refined_mask, [approx], -1, (255), -1)
    
    refined_mask = cv2.GaussianBlur(refined_mask, (5,5), 0)
    refined_mask = (refined_mask > 127).astype(np.uint8) * 255
    
    return refined_mask

def segment_banana(image, conf_threshold=0.5):
    results = model(image, conf=conf_threshold)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for r in results:
        if r.masks is None:
            continue
        
        for seg, box in zip(r.masks.data, r.boxes.data):
            cls = int(box[-1])
            if model.names[cls].lower() == 'banana':
                seg_mask = seg.cpu().numpy()
                seg_mask = cv2.resize(seg_mask, (image.shape[1], image.shape[0]))
                mask = cv2.bitwise_or(mask, (seg_mask > 0.5).astype(np.uint8) * 255)
    
    refined_mask = postprocess_mask(mask)
    
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return image, refined_mask, result

def ripeness_factor(hsv_image, mask):
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


def process_folder(folder_path):
    images = []
    processed_images = []
    features = []
    labels = []
    error_files = []
    
    for class_idx, class_name in enumerate(sorted(os.listdir(folder_path))):
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for file in os.listdir(class_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue
                
            try:
                file_path = os.path.join(class_path, file)
                image = cv2.imread(file_path)
                
                if image is None:
                    error_files.append(file_path)
                    continue
                
                contrast_img, hsv_image = preprocess_image(file_path)
                _, mask, _ = segment_banana(contrast_img)
                
                color_feats = color_features(hsv_image)
                texture_feats = texture_features(contrast_img)
                ripeness = ripeness_factor(hsv_image, mask)
                
         
                combined_features = (
                    color_feats + 
                    [
                        texture_feats['glcm_contrast'],
                        texture_feats['glcm_homogeneity'],
                        texture_feats['glcm_energy'],
                        texture_feats['glcm_entropy'],
                        texture_feats['tamura_coarseness'],
                        texture_feats['tamura_contrast'],
                        texture_feats['tamura_directionality']
                    ] + 
                    [ripeness]  
                )
                
                images.append(image)
                processed_images.append(contrast_img)
                features.append(combined_features)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                error_files.append(file_path)
                continue
    
    if error_files:
        print("\nFailed to process following files:")
        for file in error_files:
            print(file)
    
    return np.array(processed_images), np.array(features), np.array(labels)
"""
Image processing utilities for vehicle claims
"""

import os
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ExifTags
import exifread

logger = logging.getLogger(__name__)


def validate_image(file_path: str, max_size_mb: int = 10) -> Tuple[bool, str]:
    """
    Validate uploaded image file
    
    Args:
        file_path: Path to the image file
        max_size_mb: Maximum file size in MB
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
        
        # Check file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in allowed_extensions:
            return False, f"File extension '{file_ext}' is not allowed. Allowed: {', '.join(allowed_extensions)}"
        
        # Try to open image to verify it's valid
        try:
            with Image.open(file_path) as img:
                # Check minimum resolution
                width, height = img.size
                if width < 100 or height < 100:
                    return False, f"Image resolution ({width}x{height}) is too low. Minimum: 100x100"
                
                # Check maximum resolution
                if width > 8000 or height > 8000:
                    return False, f"Image resolution ({width}x{height}) is too high. Maximum: 8000x8000"
                
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
        
        return True, "Image is valid"
        
    except Exception as e:
        logger.error(f"Error validating image {file_path}: {e}")
        return False, f"Validation error: {str(e)}"


def extract_exif_data(file_path: str) -> Dict:
    """
    Extract EXIF metadata from image file
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dictionary containing EXIF data
    """
    exif_data = {}
    
    try:
        # Try using exifread first (more comprehensive)
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f)
            
            # Extract GPS coordinates
            if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                lat = _convert_to_degrees(tags['GPS GPSLatitude'].values)
                lon = _convert_to_degrees(tags['GPS GPSLongitude'].values)
                
                # Apply hemisphere
                if 'GPS GPSLatitudeRef' in tags and tags['GPS GPSLatitudeRef'].values == 'S':
                    lat = -lat
                if 'GPS GPSLongitudeRef' in tags and tags['GPS GPSLongitudeRef'].values == 'W':
                    lon = -lon
                
                exif_data['gps_latitude'] = lat
                exif_data['gps_longitude'] = lon
            
            # Extract timestamp
            if 'EXIF DateTimeOriginal' in tags:
                try:
                    datetime_str = str(tags['EXIF DateTimeOriginal'].values)
                    exif_data['datetime'] = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
                except ValueError:
                    pass
            
            # Extract camera information
            if 'Image Model' in tags:
                exif_data['camera_model'] = str(tags['Image Model'].values)
            if 'Image Make' in tags:
                exif_data['camera_make'] = str(tags['Image Make'].values)
            
            # Extract image dimensions
            if 'EXIF ExifImageWidth' in tags and 'EXIF ExifImageLength' in tags:
                exif_data['original_width'] = int(tags['EXIF ExifImageWidth'].values)
                exif_data['original_height'] = int(tags['EXIF ExifImageLength'].values)
        
        # Fallback to PIL if exifread didn't get everything
        if not exif_data.get('gps_latitude'):
            with Image.open(file_path) as img:
                exif = img._getexif()
                if exif:
                    for tag_id in exif:
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        data = exif.get(tag_id)
                        
                        if tag == 'GPSInfo':
                            gps_data = _extract_gps_from_pil(data)
                            if gps_data:
                                exif_data.update(gps_data)
                        
                        elif tag == 'DateTimeOriginal':
                            try:
                                exif_data['datetime'] = datetime.strptime(data, '%Y:%m:%d %H:%M:%S')
                            except ValueError:
                                pass
        
        logger.info(f"Extracted EXIF data from {file_path}: {exif_data}")
        return exif_data
        
    except Exception as e:
        logger.error(f"Error extracting EXIF data from {file_path}: {e}")
        return {}


def _convert_to_degrees(values) -> float:
    """Convert GPS coordinates from degrees/minutes/seconds to decimal degrees"""
    try:
        degrees = float(values[0].num) / float(values[0].den)
        minutes = float(values[1].num) / float(values[1].den)
        seconds = float(values[2].num) / float(values[2].den)
        
        return degrees + (minutes / 60.0) + (seconds / 3600.0)
    except (IndexError, AttributeError, ZeroDivisionError):
        return 0.0


def _extract_gps_from_pil(gps_data: Dict) -> Optional[Dict]:
    """Extract GPS coordinates from PIL EXIF data"""
    try:
        if not gps_data:
            return None
        
        lat = _get_gps_coordinate(gps_data, 1, 2, 3, 'S')
        lon = _get_gps_coordinate(gps_data, 3, 4, 5, 'W')
        
        if lat is not None and lon is not None:
            return {
                'gps_latitude': lat,
                'gps_longitude': lon
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting GPS from PIL data: {e}")
        return None


def _get_gps_coordinate(gps_data: Dict, ref_key: int, lat_key: int, lon_key: int, negative_ref: str) -> Optional[float]:
    """Get GPS coordinate from PIL GPS data"""
    try:
        if ref_key in gps_data and lat_key in gps_data and lon_key in gps_data:
            ref = gps_data[ref_key]
            lat = gps_data[lat_key]
            lon = gps_data[lon_key]
            
            if ref == negative_ref:
                return -(_convert_gps_to_degrees(lat, lon))
            else:
                return _convert_gps_to_degrees(lat, lon)
        
        return None
        
    except Exception:
        return None


def _convert_gps_to_degrees(lat_data: Tuple, lon_data: Tuple) -> float:
    """Convert GPS data tuple to decimal degrees"""
    try:
        degrees = float(lat_data[0]) / float(lat_data[1])
        minutes = float(lat_data[2]) / float(lat_data[3])
        seconds = float(lat_data[4]) / float(lat_data[5])
        
        return degrees + (minutes / 60.0) + (seconds / 3600.0)
    except (IndexError, ZeroDivisionError):
        return 0.0


def preprocess_image_for_ai(image_path: str, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Preprocess image for AI model input
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the model (width, height)
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        raise


def create_visualization_image(original_image_path: str, detected_parts: list, detected_damages: list) -> np.ndarray:
    """
    Create visualization image with bounding boxes and annotations
    
    Args:
        original_image_path: Path to original image
        detected_parts: List of detected parts with bounding boxes
        detected_damages: List of detected damages with bounding boxes
        
    Returns:
        Visualization image as numpy array
    """
    try:
        # Load original image
        image = cv2.imread(original_image_path)
        if image is None:
            raise ValueError("Failed to load original image")
        
        # Draw bounding boxes for detected parts
        for part in detected_parts:
            if hasattr(part, 'bbox') and part.bbox:
                x1, y1, x2, y2 = map(int, part.bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{part.part_name} ({part.confidence_score:.2f})", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw bounding boxes for detected damages
        for damage in detected_damages:
            if hasattr(damage, 'bbox') and damage.bbox:
                x1, y1, x2, y2 = map(int, damage.bbox)
                # Use red color for damages
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image, f"{damage.damage_type} ({damage.severity_level})", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return image
        
    except Exception as e:
        logger.error(f"Error creating visualization image: {e}")
        raise


def save_processed_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save processed image to file
    
    Args:
        image: Image as numpy array
        output_path: Path to save the image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert back to uint8 if needed
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        
        # Save image
        cv2.imwrite(output_path, image)
        
        logger.info(f"Saved processed image to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving processed image to {output_path}: {e}")
        return False


def calculate_image_quality_score(image_path: str) -> float:
    """
    Calculate image quality score based on various metrics
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Quality score between 0 and 1
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0.0
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalize to [0, 1]
        
        # Calculate brightness score
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128  # Closer to 128 is better
        
        # Calculate contrast score
        contrast = np.std(gray)
        contrast_score = min(contrast / 50.0, 1.0)  # Normalize to [0, 1]
        
        # Combine scores (weighted average)
        quality_score = (sharpness_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)
        
        return max(0.0, min(1.0, quality_score))
        
    except Exception as e:
        logger.error(f"Error calculating image quality score: {e}")
        return 0.0


def detect_blur(image_path: str, threshold: float = 100.0) -> Tuple[bool, float]:
    """
    Detect if image is blurry
    
    Args:
        image_path: Path to the image file
        threshold: Blur threshold (lower = more sensitive)
        
    Returns:
        Tuple of (is_blurry, blur_score)
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return True, 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        is_blurry = laplacian_var < threshold
        blur_score = max(0.0, 1.0 - (laplacian_var / threshold))
        
        return is_blurry, blur_score
        
    except Exception as e:
        logger.error(f"Error detecting blur: {e}")
        return True, 1.0 
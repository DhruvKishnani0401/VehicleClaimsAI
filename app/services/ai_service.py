"""
AI Service for Vehicle Claims Processing
Handles VLM and CNN-based damage assessment
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from ultralytics import YOLO
from openai import AzureOpenAI
import cv2

from app.models.damage_assessment import DamageAssessment, DetectedPart, DetectedDamage
from app.utils.image_processing import extract_exif_data, validate_image

logger = logging.getLogger(__name__)

@dataclass
class AIModelConfig:
    """Configuration for AI models"""
    confidence_threshold: float = 0.7
    damage_severity_thresholds: Dict[str, float] = None
    base_repair_costs: Dict[str, float] = None
    severity_multipliers: Dict[str, float] = None

class VehicleDamageAIService:
    """Main AI service for vehicle damage assessment"""
    
    def __init__(self, config: AIModelConfig):
        self.config = config
        self.vlm_client = None
        self.vehicle_parts_model = None
        self.damage_detection_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models"""
        try:
            # Initialize Azure OpenAI client for VLM
            self.vlm_client = AzureOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version=os.getenv("OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            
            # Initialize YOLO model for vehicle parts detection
            self.vehicle_parts_model = YOLO('yolov8n.pt')  # Start with base model
            
            # Initialize custom damage detection model (placeholder)
            self.damage_detection_model = self._load_damage_model()
            
            logger.info("AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            raise
    
    def _load_damage_model(self):
        """Load custom damage detection model"""
        # TODO: Implement custom damage detection model
        # For now, return a placeholder
        return None
    
    async def process_claim_image(self, image_path: str, claim_data: Dict) -> DamageAssessment:
        """
        Main method to process a claim image and generate damage assessment
        
        Args:
            image_path: Path to the uploaded image
            claim_data: Claim metadata including location, timestamp
            
        Returns:
            DamageAssessment object with all analysis results
        """
        try:
            # Validate and preprocess image
            image = await self._preprocess_image(image_path)
            
            # Extract EXIF data
            exif_data = extract_exif_data(image_path)
            
            # Parallel processing tasks
            tasks = [
                self._analyze_with_vlm(image),
                self._detect_vehicle_parts(image),
                self._detect_damages(image),
                self._verify_location(exif_data, claim_data),
                self._check_fraud_indicators(image, exif_data)
            ]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Unpack results
            vlm_analysis = results[0] if not isinstance(results[0], Exception) else None
            vehicle_parts = results[1] if not isinstance(results[1], Exception) else []
            damages = results[2] if not isinstance(results[2], Exception) else []
            location_verification = results[3] if not isinstance(results[3], Exception) else {}
            fraud_indicators = results[4] if not isinstance(results[4], Exception) else []
            
            # Generate cost estimate
            cost_estimate = self._estimate_repair_cost(damages, vehicle_parts)
            
            # Create comprehensive assessment
            assessment = DamageAssessment(
                image_path=image_path,
                vlm_summary=vlm_analysis.get('summary') if vlm_analysis else None,
                vlm_anomalies=vlm_analysis.get('anomalies') if vlm_analysis else None,
                detected_parts=vehicle_parts,
                detected_damages=damages,
                estimated_cost=cost_estimate,
                location_verification=location_verification,
                fraud_indicators=fraud_indicators,
                confidence_score=self._calculate_overall_confidence(vehicle_parts, damages)
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error processing claim image: {e}")
            raise
    
    async def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for AI analysis"""
        try:
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Invalid image file")
            
            # Resize to standard size for model input
            image = cv2.resize(image, (640, 640))
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    async def _analyze_with_vlm(self, image: np.ndarray) -> Dict:
        """Analyze image using Visual Language Model (Azure OpenAI GPT-4 Vision)"""
        try:
            # Convert numpy array to PIL Image for OpenAI
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Prepare prompt for damage assessment
            prompt = """
            Analyze this vehicle image for insurance claims processing. Please provide:
            1. Vehicle description (type, color, make/model if visible)
            2. Overall damage assessment summary
            3. Any visual anomalies or suspicious elements
            4. General condition assessment
            
            Focus on identifying damage types, severity, and any potential fraud indicators.
            """
            
            # Call Azure OpenAI Vision API
            response = self.vlm_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self._encode_image(pil_image)}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            # Parse response
            analysis_text = response.choices[0].message.content
            
            # Extract key information (simplified parsing)
            return {
                'summary': analysis_text,
                'anomalies': self._extract_anomalies(analysis_text),
                'vehicle_info': self._extract_vehicle_info(analysis_text)
            }
            
        except Exception as e:
            logger.error(f"Error in VLM analysis: {e}")
            return {'summary': 'Analysis failed', 'anomalies': [], 'vehicle_info': {}}
    
    async def _detect_vehicle_parts(self, image: np.ndarray) -> List[DetectedPart]:
        """Detect vehicle parts using YOLO model"""
        try:
            # Run YOLO inference
            results = self.vehicle_parts_model(image)
            
            detected_parts = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Filter by confidence threshold
                        if box.conf[0] > self.config.confidence_threshold:
                            # Map YOLO classes to vehicle parts
                            part_name = self._map_yolo_class_to_part(int(box.cls[0]))
                            
                            detected_part = DetectedPart(
                                part_name=part_name,
                                bbox=box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                                confidence_score=float(box.conf[0])
                            )
                            detected_parts.append(detected_part)
            
            return detected_parts
            
        except Exception as e:
            logger.error(f"Error detecting vehicle parts: {e}")
            return []
    
    async def _detect_damages(self, image: np.ndarray) -> List[DetectedDamage]:
        """Detect damages using custom CNN model"""
        try:
            # TODO: Implement custom damage detection model
            # For now, return placeholder results
            damages = []
            
            # Placeholder damage detection logic
            # This would be replaced with actual CNN inference
            
            return damages
            
        except Exception as e:
            logger.error(f"Error detecting damages: {e}")
            return []
    
    async def _verify_location(self, exif_data: Dict, claim_data: Dict) -> Dict:
        """Verify image location against reported claim location"""
        try:
            verification_result = {
                'gps_match': True,
                'timestamp_match': True,
                'distance_km': 0.0,
                'time_difference_hours': 0.0
            }
            
            if 'gps_latitude' in exif_data and 'gps_longitude' in exif_data:
                # Calculate distance between image GPS and reported location
                image_lat = exif_data['gps_latitude']
                image_lon = exif_data['gps_longitude']
                reported_lat = claim_data.get('reported_latitude')
                reported_lon = claim_data.get('reported_longitude')
                
                if reported_lat and reported_lon:
                    distance = self._calculate_distance(
                        image_lat, image_lon, reported_lat, reported_lon
                    )
                    verification_result['distance_km'] = distance
                    verification_result['gps_match'] = distance <= float(os.getenv('GPS_MISMATCH_THRESHOLD_KM', 5.0))
            
            # Check timestamp consistency
            if 'datetime' in exif_data and 'incident_time' in claim_data:
                time_diff = abs(
                    exif_data['datetime'] - claim_data['incident_time']
                ).total_seconds() / 3600
                verification_result['time_difference_hours'] = time_diff
                verification_result['timestamp_match'] = time_diff <= float(os.getenv('TIMESTAMP_MISMATCH_HOURS', 2.0))
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Error verifying location: {e}")
            return {'gps_match': False, 'timestamp_match': False, 'distance_km': 0.0, 'time_difference_hours': 0.0}
    
    async def _check_fraud_indicators(self, image: np.ndarray, exif_data: Dict) -> List[Dict]:
        """Check for potential fraud indicators"""
        fraud_indicators = []
        
        try:
            # Check for missing or manipulated EXIF data
            if not exif_data.get('gps_latitude') or not exif_data.get('gps_longitude'):
                fraud_indicators.append({
                    'type': 'missing_gps',
                    'confidence': 0.8,
                    'description': 'GPS coordinates missing from image metadata'
                })
            
            # Check for image manipulation indicators
            manipulation_score = self._detect_image_manipulation(image)
            if manipulation_score > 0.7:
                fraud_indicators.append({
                    'type': 'image_manipulation',
                    'confidence': manipulation_score,
                    'description': 'Potential image manipulation detected'
                })
            
            return fraud_indicators
            
        except Exception as e:
            logger.error(f"Error checking fraud indicators: {e}")
            return fraud_indicators
    
    def _estimate_repair_cost(self, damages: List[DetectedDamage], parts: List[DetectedPart]) -> float:
        """Estimate repair cost based on detected damages and parts"""
        try:
            total_cost = 0.0
            
            for damage in damages:
                # Get base cost for the damaged part
                base_cost = self.config.base_repair_costs.get(damage.associated_part, 500)
                
                # Apply severity multiplier
                severity_multiplier = self.config.severity_multipliers.get(damage.severity_level, 1.0)
                
                # Calculate cost for this damage
                damage_cost = base_cost * severity_multiplier
                total_cost += damage_cost
            
            return round(total_cost, 2)
            
        except Exception as e:
            logger.error(f"Error estimating repair cost: {e}")
            return 0.0
    
    def _calculate_overall_confidence(self, parts: List[DetectedPart], damages: List[DetectedDamage]) -> float:
        """Calculate overall confidence score for the assessment"""
        try:
            if not parts and not damages:
                return 0.0
            
            total_confidence = 0.0
            total_items = 0
            
            for part in parts:
                total_confidence += part.confidence_score
                total_items += 1
            
            for damage in damages:
                total_confidence += damage.confidence_score
                total_items += 1
            
            return total_confidence / total_items if total_items > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    # Helper methods
    def _map_yolo_class_to_part(self, class_id: int) -> str:
        """Map YOLO class ID to vehicle part name"""
        # Simplified mapping - would need to be customized based on training data
        part_mapping = {
            0: 'bumper', 1: 'hood', 2: 'door', 3: 'fender', 4: 'windshield'
        }
        return part_mapping.get(class_id, 'unknown')
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64 string"""
        import base64
        import io
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in kilometers"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r
    
    def _detect_image_manipulation(self, image: np.ndarray) -> float:
        """Detect potential image manipulation"""
        # TODO: Implement image manipulation detection
        # This could use techniques like Error Level Analysis (ELA)
        # For now, return a placeholder score
        return 0.1
    
    def _extract_anomalies(self, analysis_text: str) -> List[str]:
        """Extract anomalies from VLM analysis text"""
        # Simple keyword-based extraction
        anomaly_keywords = ['suspicious', 'anomaly', 'manipulated', 'inconsistent', 'unusual']
        anomalies = []
        
        for keyword in anomaly_keywords:
            if keyword.lower() in analysis_text.lower():
                anomalies.append(f"Potential {keyword} detected")
        
        return anomalies
    
    def _extract_vehicle_info(self, analysis_text: str) -> Dict:
        """Extract vehicle information from VLM analysis"""
        # Simple extraction - could be enhanced with more sophisticated parsing
        return {
            'description': analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text
        } 
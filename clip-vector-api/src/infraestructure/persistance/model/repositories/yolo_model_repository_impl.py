"""
YOLO Model Repository Implementation - Infrastructure Layer
"""
import logging
import time
from typing import Optional
from io import BytesIO

from PIL import Image

from src.domain.model.repositories.yolo_model_repository import YoloModelRepository
from src.domain.model.models.detection_models import DetectionResult, BoundingBox
from src.infraestructure.persistance.model.config.yolo_config import YoloModelConfig


logger = logging.getLogger(__name__)


class YoloModelRepositoryImpl(YoloModelRepository):
    """
    Real implementation of YOLO service using ultralytics YOLOv8.
    Follows the repository pattern with synchronous and asynchronous methods.
    Optimized for CPU-only inference.
    """
    
    def __init__(self, config: Optional[YoloModelConfig] = None):
        """
        Initialize YOLO model repository.
        
        Args:
            config: YOLO model configuration. If None, loads from environment.
        """
        self.config = config or YoloModelConfig.from_env()
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load YOLOv8 model with optimized configuration.
        Uses CPU-only inference as configured.
        """
        try:
            from ultralytics import YOLO # type: ignore
            
            logger.info(f"Loading YOLO model: {self.config.model_version}...")
            
            # Load YOLOv8 model (will download on first use)
            # Model files are automatically downloaded to ~/.cache/ultralytics/
            self.model = YOLO(f"{self.config.model_version}.pt")
            
            # Set device to CPU
            self.model.to(self.config.device)
            
            logger.info(f"✅ YOLO model {self.config.model_version} loaded successfully on device: {self.config.device}")
            logger.info(f"📊 Confidence threshold: {self.config.confidence_threshold}, IOU threshold: {self.config.iou_threshold}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            logger.error("Make sure ultralytics is installed: pip install ultralytics")
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    async def detect_objects(self, image_data: BytesIO) -> DetectionResult:
        """
        Detect objects in an image using YOLO model (async).
        Async wrapper that calls the sync implementation.
        
        Args:
            image_data: The image data as BytesIO
            
        Returns:
            DetectionResult containing all detected objects
        """
        return self.detect_objects_sync(image_data)
    
    def detect_objects_sync(self, image_data: BytesIO) -> DetectionResult:
        """
        Detect objects in an image using YOLO model (synchronous).
        
        Args:
            image_data: The image data as BytesIO
            
        Returns:
            DetectionResult containing all detected objects with bounding boxes
        """
        try:
            # Reset stream position
            image_data.seek(0)
            
            if self.model is None:
                raise RuntimeError("YOLO model not available")
            
            # Load image
            image = Image.open(image_data)
            
            # Convert to RGB if necessary
            image = self._ensure_image_rgb(image)
            
            # Get image dimensions
            image_width, image_height = image.size
            
            # Start timing
            start_time = time.time()
            
            # Run YOLO detection
            results = self.model(
                image,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                verbose=False,  # Suppress YOLO output
                device=self.config.device,
            )
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Parse results
            bounding_boxes = self._parse_yolo_results(results)
            
            logger.info(f"Detected {len(bounding_boxes)} objects in {inference_time:.2f}ms")
            
            # Create DetectionResult
            detection_result = DetectionResult(
                bounding_boxes=bounding_boxes,
                image_width=image_width,
                image_height=image_height,
                inference_time_ms=inference_time,
            )
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Error checking YOLO detection: {e}")
            raise RuntimeError(f"Failed to detect objects: {e}")

    def _ensure_image_rgb(self, image: Image.Image) -> Image.Image:
        """Ensure image is in RGB format, handling transparency properly."""
        if image.mode == 'RGB':
            return image
            
        if image.mode == 'RGBA':
            # Handle transparency by compositing on white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
            return background
            
        return image.convert('RGB')

    def _parse_yolo_results(self, results) -> list[BoundingBox]:
        """Parse YOLO results into BoundingBox objects."""
        bounding_boxes = []
        
        if not results or len(results) == 0:
            return bounding_boxes
            
        result = results[0]  # First image (we only process one)
        
        if result.boxes is None:
            return bounding_boxes
            
        boxes = result.boxes
        
        # Extract bounding boxes
        for i in range(len(boxes)):
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            
            # Get confidence
            confidence = float(boxes.conf[i])
            
            # Get class ID and name
            class_id = int(boxes.cls[i])
            class_name = self.model.names[class_id]
            
            # Create BoundingBox
            bbox = BoundingBox(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            )
            bounding_boxes.append(bbox)
            
        return bounding_boxes

    
    def crop_object(self, image: Image.Image, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
        """
        Crop an object from an image using bounding box coordinates.
        
        Args:
            image: The source image
            x1: Top-left x coordinate
            y1: Top-left y coordinate
            x2: Bottom-right x coordinate
            y2: Bottom-right y coordinate
            
        Returns:
            Cropped image of the detected object
        """
        try:
            # Validate coordinates
            if x1 < 0 or y1 < 0 or x2 > image.width or y2 > image.height:
                logger.warning(f"Bounding box coordinates out of bounds: ({x1},{y1},{x2},{y2}), image size: {image.size}")
                # Clip coordinates to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.width, x2)
                y2 = min(image.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid bounding box: ({x1},{y1},{x2},{y2})")
            
            # Crop image
            cropped = image.crop((x1, y1, x2, y2))
            
            logger.debug(f"Cropped object: bbox=({x1},{y1},{x2},{y2}), size={cropped.size}")
            
            return cropped
            
        except Exception as e:
            logger.error(f"Error cropping object: {e}")
            raise RuntimeError(f"Failed to crop object: {e}")

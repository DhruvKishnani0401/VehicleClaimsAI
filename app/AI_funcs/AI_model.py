from roboflow import Roboflow
import os
from fastapi import HTTPException

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

async def detect_damages(image_path):
    parsed_parts = []
    damage_part_model = rf.workspace().project("car-damage-detection-5ioys").version(1).model

    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail="Image file missing on server")

    part_pred = damage_part_model.predict(image_path).json()

    for pred in part_pred.get("predictions", []):
        x1, y1 = pred["x"], pred["y"]
        x2, y2 = x1 + pred["width"], y1 + pred["height"]
        parsed_parts.append({
            "part_name": pred.get("class", ""),
            "bbox": [x1, y1, x2, y2],
            "confidence_score": pred.get("confidence", 0)
        })

    return parsed_parts

async def detect_types(image_path):
    parsed_type = []
    damage_type_model = rf.workspace().project("damage-type-cjmf5").version(2).model
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail="Image file missing on server")

    type_pred = damage_type_model.predict(image_path).json()

    for pred in type_pred.get("predictions", []):
        x1, y1 = pred["x"], pred["y"]
        x2, y2 = x1 + pred["width"], y1 + pred["height"]
        parsed_type.append({
            "damage_type": pred.get("class", ""),
            "severity_level": pred.get("severity", "unknown"),
            "confidence_score": pred.get("confidence", 0),
            "bbox": [x1, y1, x2, y2],
            "associated_part": None
        })

    return parsed_type

#damage_severity_model = rf.workspace().project("car-damage-severity-assessment").version(1).model
#severity_pred = damage_severity_model.predict(image_path).json()
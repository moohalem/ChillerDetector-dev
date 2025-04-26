import os
import sys
import json
from ultralytics import YOLO

# Initialize the model once
model = YOLO("weights/chiller_cls_prod.pt")

def main(image_path: str) -> dict:
    """
    Classify an image as 'Chiller' or 'Non Chiller'.

    Args:
        image_path (str): Path to the input image file (.jpg, .jpeg).

    Returns:
        dict: A dictionary with status, prediction, and confidence or error details.
    """
    try:
        # File and path validation
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File {image_path} is not found")
        if not image_path.lower().endswith(('.jpg', '.jpeg')):
            raise ValueError("File must have .jpg or .jpeg extension")

        # Run prediction
        prediction = model(image_path)[0]
        names = prediction.names           # e.g., {0: 'Chiller', 1: 'Non Chiller'}
        probs = prediction.probs.data.tolist()

        # Identify best class
        max_index = probs.index(max(probs))
        best_class = names[max_index]
        best_prob = probs[max_index]

        return {
            "status": "success",
            "prediction": best_class,
            "confidence": round(float(best_prob), 3)
        }

    except Exception as e:
        return {
            "status": "error",
            "image_path": image_path,
            "error": str(e)
        }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({
            "status": "error",
            "message": "Usage: python chiller_cls.py <path_to_image>"
        }))
        sys.exit(1)

    img_path = sys.argv[1]
    result = main(img_path)
    print(json.dumps(result))

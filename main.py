import io

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image

from pipeline import ChillerBrandPipeline

# Instantiate your pipeline
pipeline = ChillerBrandPipeline(
    chiller_cls_model_path="weights/chiller_cls_prod.pt",
    shape_seg_model_path="weights/shape_seg_prod.pt",
    brand_cls_model_path="weights/brand_cls_prod.pt",
)

app = FastAPI()


def read_imagefile(file) -> np.ndarray:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return np.array(image)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_np = read_imagefile(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Step 1: Check if it's a chiller photo
    is_chiller = await run_in_threadpool(pipeline.predict_chiller, image_np)
    if not is_chiller:
        return JSONResponse(
            status_code=400,
            content={
                "detail": "Not a photo of a chiller. Please upload a correct image."
            },
        )

    # Step 2-3: Segment + Get Cans
    cans = await run_in_threadpool(pipeline.seg_to_bboxes, image_np)
    pipeline.save_json({"cans": cans}, "cans.json")

    # Step 4: Filter Large Bounding Boxes
    filtered = pipeline.filter_large(cans, min_area=5000)  # Example threshold: 5000 px²
    pipeline.save_json({"brands": filtered}, "brands.json")

    # Step 5: Crop brands
    crops = pipeline.crop_and_save(image_np, filtered)

    # Step 6: Classify brands
    brand_preds = pipeline.classify_brands(crops)

    # Step 7: Update brands.json
    updated_brands = pipeline.update_brands_json(filtered, brand_preds)
    pipeline.save_json({"brands": updated_brands}, "brands.json")

    # Step 8: Run CanCounter
    counts = pipeline.run_counter(cans, updated_brands)

    return {"cans": cans, "brands": updated_brands, "counts": counts}


if __name__ == "__main__":
    uvicorn.run("main:pipeline", host="127.0.0.1", port=8000, reload=True)

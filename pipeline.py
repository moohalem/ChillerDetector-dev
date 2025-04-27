import json
import math
import os
import uuid
from typing import Dict, List

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from ultralytics import YOLO

# Configuration for can-lid grouping
PRODUCT_HOR_MULTIPLIER = 0.3
PRODUCT_VERT_MULTIPLIER = 0.3
CL0_HOR_MULTIPLIER = 0.1
VERT_VARIATION = 0.3
DYNAMIC_CL0_BASE = 80
SLOPE = 0.05


# Helper functions for grouping
##########################################################################################################
def get_dynamic_multiplier(avg_height):
    multiplier = 1.5 - SLOPE * (avg_height - DYNAMIC_CL0_BASE)
    return max(multiplier, 1.5)


def euclidean_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def compute_average_dimensions(lids):
    if not lids:
        return 0.0, 0.0
    widths = [c["x2"] - c["x1"] for c in lids]
    heights = [c["y2"] - c["y1"] for c in lids]
    return sum(widths) / len(widths), sum(heights) / len(heights)


def fit_line_sklearn(points):
    if len(points) < 2:
        if points:
            return 0.0, points[0][1]
        return 0.0, 0.0
    xs, ys = zip(*points)
    if min(xs) == max(xs):
        return None
    X = np.array(xs).reshape(-1, 1)
    y = np.array(ys)
    model = LinearRegression().fit(X, y)
    return model.coef_[0], model.intercept_


def draw_box(image, x1, y1, x2, y2, color, thickness=2, label=None):
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    if label:
        cv2.putText(
            image,
            label,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )


########################################################################################################


class CanCounter:
    """
    Groups can-lid detections to count cans associated with each product.
    Expects jsondata containing 'product' and 'canlids' predictions.
    """

    def __init__(self):
        self.canlids = []
        self.avg_width = 0.0
        self.avg_height = 0.0

    def find_cl0(self, product):
        dynamic = get_dynamic_multiplier(self.avg_height)
        top_cx, top_y = product["center_x"], product["y1"]
        best, mind = None, float("inf")
        # first try inside boundary
        for c in self.canlids:
            if (
                product["x1"] <= c["center_x"] <= product["x2"]
                and c["center_y"] <= product["center_y"]
            ):
                d = euclidean_distance(top_cx, top_y, c["center_x"], c["center_y"])
                if d < mind:
                    mind, best = d, c
        if best:
            return best
        # search with multipliers
        for c in self.canlids:
            if (
                not c["assigned"]
                and c["center_y"] <= top_y + PRODUCT_VERT_MULTIPLIER * self.avg_height
            ):
                if (
                    product["x1"] - PRODUCT_HOR_MULTIPLIER * self.avg_width
                    <= c["center_x"]
                    <= product["x2"] + PRODUCT_HOR_MULTIPLIER * self.avg_width
                ):
                    vert = top_y - c["center_y"]
                    if vert <= dynamic * self.avg_height:
                        d = euclidean_distance(
                            top_cx, top_y, c["center_x"], c["center_y"]
                        )
                        if d < mind:
                            mind, best = d, c
        return best

    def find_cl1(self, cl0):
        if cl0 is None:
            return None
        dynamic = get_dynamic_multiplier(self.avg_height)
        best, mind = None, float("inf")
        for c in self.canlids:
            if c["assigned"]:
                continue
            if c["center_y"] < cl0["center_y"]:
                if (
                    cl0["x1"] - CL0_HOR_MULTIPLIER * self.avg_width
                    <= c["center_x"]
                    <= cl0["x2"] + CL0_HOR_MULTIPLIER * self.avg_width
                ):
                    d = euclidean_distance(
                        cl0["center_x"], cl0["center_y"], c["center_x"], c["center_y"]
                    )
                    if d < dynamic * self.avg_height and d < mind:
                        mind, best = d, c
        return best

    def iterative_grouping(self, assigned):
        while True:
            last = assigned[-1]
            nxt = self.find_cl1(last)
            if not nxt:
                break
            nxt["assigned"] = True
            assigned.append(nxt)

    def count(self, jsondata, image_path=None, is_debug=False):
        # parse products and lids
        products = []
        for p in jsondata.get("product", {}).get("prediction", []):
            b = p["box"]
            products.append(
                {
                    "name": p["name"],
                    "x1": b["x1"],
                    "y1": b["y1"],
                    "x2": b["x2"],
                    "y2": b["y2"],
                    "center_x": (b["x1"] + b["x2"]) / 2,
                    "center_y": (b["y1"] + b["y2"]) / 2,
                }
            )
        self.canlids = []
        for c in jsondata.get("canlids", {}).get("prediction", []):
            b = c["box"]
            self.canlids.append(
                {
                    "name": c["name"],
                    "x1": b["x1"],
                    "y1": b["y1"],
                    "x2": b["x2"],
                    "y2": b["y2"],
                    "center_x": (b["x1"] + b["x2"]) / 2,
                    "center_y": (b["y1"] + b["y2"]) / 2,
                    "assigned": False,
                }
            )
        # averages
        self.avg_width, self.avg_height = compute_average_dimensions(self.canlids)
        # sort products
        threshold = self.avg_height * VERT_VARIATION
        products.sort(key=lambda p: (round(p["y1"] / threshold) * threshold, p["x1"]))
        # assign lids
        assignments = []
        for prod in products:
            cl0 = self.find_cl0(prod)
            if not cl0:
                assignments.append(
                    {"name": prod["name"], "assigned": "NONE", "product": prod}
                )
                continue
            cl0["assigned"] = True
            assigned = [cl0]
            cl1 = self.find_cl1(cl0)
            if cl1:
                cl1["assigned"] = True
                assigned.append(cl1)
            self.iterative_grouping(assigned)
            # Build lids_info with clear variable name
            lids_info = [
                {
                    "x1": lid["x1"],
                    "y1": lid["y1"],
                    "x2": lid["x2"],
                    "y2": lid["y2"],
                    "center_x": lid["center_x"],
                    "center_y": lid["center_y"],
                }
                for lid in assigned
            ]
            assignments.append(
                {"name": prod["name"], "assigned": lids_info, "product": prod}
            )
        # debug draw
        if is_debug and image_path:
            self._draw_debug(image_path, assignments, products)
        # count
        counts = {}
        for entry in assignments:
            key = entry["name"]
            counts[key] = counts.get(key, 0) + (
                len(entry["assigned"]) if entry["assigned"] != "NONE" else 1
            )
        return counts

    def _draw_debug(self, image_path, assignments, products):
        img = cv2.imread(image_path)
        if img is None:
            return False
        for p in products:
            draw_box(img, p["x1"], p["y1"], p["x2"], p["y2"], (0, 0, 255), 1)
        for entry in assignments:
            color = (
                int(hash(entry["name"]) % 256),
                int(hash(entry["name"] + "1") % 256),
                int(hash(entry["name"] + "2") % 256),
            )
            if entry["assigned"] == "NONE":
                p = entry["product"]
                cv2.putText(
                    img,
                    f"{p['name']}:NONE",
                    (int(p["x1"]), int(p["y1"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            else:
                for idx, lid in enumerate(entry["assigned"]):
                    draw_box(
                        img,
                        lid["x1"],
                        lid["y1"],
                        lid["x2"],
                        lid["y2"],
                        color,
                        2,
                        f"{entry['name']}_{idx + 1}",
                    )
        debug_path = os.path.join("output", "debug_" + os.path.basename(image_path))
        cv2.imwrite(debug_path, img)
        return True


class ChillerBrandPipeline:
    def __init__(
        self, chiller_cls_model_path, shape_seg_model_path, brand_cls_model_path
    ):
        self.chiller_model = YOLO(chiller_cls_model_path)
        self.seg_model = YOLO(shape_seg_model_path)
        self.brand_model = YOLO(brand_cls_model_path)
        self.can_counter = CanCounter()

    def predict_chiller(self, image_np: np.ndarray) -> bool:
        result = self.chiller_model.predict(source=image_np, imgsz=224)[0]
        pred_idx = int(result.probs.top1)
        pred_name = self.chiller_model.names[pred_idx]
        return pred_name.lower() == "chiller"

    def seg_to_bboxes(self, image_np: np.ndarray, conf=0.25) -> List[Dict]:
        result = self.seg_model.predict(source=image_np, conf=conf, task="segment")[0]
        boxes = []
        for box, cls_id, score in zip(
            result.boxes.xyxy, result.boxes.cls, result.boxes.conf
        ):
            x1, y1, x2, y2 = map(int, box.tolist())
            boxes.append(
                {
                    "id": str(uuid.uuid4()),
                    "class_id": int(cls_id),
                    "score": float(score),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )
        return boxes

    def filter_large(self, boxes: List[Dict], min_area: int) -> List[Dict]:
        large = []
        for box in boxes:
            coords = box["box"]
            area = (coords["x2"] - coords["x1"]) * (coords["y2"] - coords["y1"])
            if area >= min_area:
                large.append(box)
        return large

    def crop_and_save(
        self, image_np: np.ndarray, boxes: List[Dict], out_dir="temp_cropped"
    ) -> List[Dict]:
        os.makedirs(out_dir, exist_ok=True)
        crops = []
        for box in boxes:
            coords = box["box"]
            crop = image_np[coords["y1"] : coords["y2"], coords["x1"] : coords["x2"]]
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(out_dir, filename)
            cv2.imwrite(filepath, crop[..., ::-1])  # convert RGB to BGR
            crops.append({"id": box["id"], "path": filepath})
        return crops

    def classify_brands(self, crops: List[Dict], conf=0.5) -> List[Dict]:
        predictions = []
        for crop in crops:
            res = self.brand_model.predict(source=crop["path"], conf=conf, imgsz=224)[0]
            idx = int(res.probs.top1)
            label = self.brand_model.names[idx]
            predictions.append({"id": crop["id"], "brand": label})
        return predictions

    def update_brands_json(self, brands: List[Dict], preds: List[Dict]) -> List[Dict]:
        pred_map = {pred["id"]: pred["brand"] for pred in preds}
        for brand in brands:
            if brand["id"] in pred_map:
                brand["brand"] = pred_map[brand["id"]]
        return brands

    def save_json(self, data: Dict, path: str):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_json(self, path: str) -> Dict:
        with open(path, "r") as f:
            return json.load(f)

    def run_counter(self, cans: List[Dict], brands: List[Dict]) -> Dict:
        data = {"product": {"prediction": brands}, "canlids": {"prediction": cans}}
        return self.can_counter.count(data, is_debug=False)

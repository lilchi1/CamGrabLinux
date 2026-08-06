# main_working.py
import sys
import numpy as np
import cv2
import tensorflow as tf
import os

def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)

# Якоря YOLOv2
ANCHORS = np.array([
    [18.32736, 21.67632],
    [59.98272, 66.00096],
    [106.82976, 175.17888],
    [252.25024, 112.88896],
    [312.65664, 293.38496],
], dtype=np.float32)

CONF_THRESH = 0.30
NMS_THRESH = 0.45

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))

def decode_yolov2(pred, anchors, conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH):
    """Декодирование предсказаний YOLOv2"""
    p = pred[0]
    grid = p.shape[0]
    num_anchors = anchors.shape[0]
    num_classes = p.shape[2] // num_anchors - 5
    
    p = p.reshape(grid, grid, num_anchors, 5 + num_classes)
    stride = 608 // grid
    
    boxes = []
    for gy in range(grid):
        for gx in range(grid):
            for a in range(num_anchors):
                tx, ty, tw, th, obj = p[gy, gx, a, :5]
                classes = p[gy, gx, a, 5:]
                
                conf = sigmoid(obj)
                if conf < conf_thresh:
                    continue
                
                cls_scores = sigmoid(classes)
                cls = int(np.argmax(cls_scores))
                score = conf * cls_scores[cls]
                
                if score < conf_thresh:
                    continue
                
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a, 0] * np.exp(tw)
                h = anchors[a, 1] * np.exp(th)
                
                boxes.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2, score, cls))
    
    if not boxes:
        return []
    
    # NMS
    boxes.sort(key=lambda b: -b[4])
    keep = []
    for b in boxes:
        suppressed = False
        for k in keep:
            if b[5] != k[5]:
                continue
            ix1, iy1 = max(b[0], k[0]), max(b[1], k[1])
            ix2, iy2 = min(b[2], k[2]), min(b[3], k[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            area_k = (k[2] - k[0]) * (k[3] - k[1])
            if inter / (area_b + area_k - inter) > nms_thresh:
                suppressed = True
                break
        if not suppressed:
            keep.append(b)
    
    return keep

def main():
    # Загружаем модель
    model_path = 'yolov2_working.keras'
    
    if not os.path.exists(model_path):
        print("⚠️ Модель не найдена. Сначала запустите load_weights_direct.py")
        return
    
    print("🚀 Загрузка модели...")
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"space_to_depth_x2": space_to_depth_x2}
    )
    print("✅ Модель загружена")
    
    # Загружаем изображение
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"❌ {image_path} не найден")
        print("Создаю тестовое изображение...")
        img = np.full((608, 608, 3), 114, dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), 3)
        cv2.putText(img, "TEST", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imwrite("test.jpg", img)
        img = cv2.imread("test.jpg")
    
    # Подготовка
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (608, 608))
    x = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    
    # Предсказание
    print("🔮 Предсказание...")
    out = model.predict(x, verbose=0)
    print(f"✅ Выходной shape: {out.shape}")
    
    # Декодирование
    dets = decode_yolov2(out, ANCHORS)
    print(f"\n🔍 Найдено объектов: {len(dets)}")
    
    # Рисуем
    for x1, y1, x2, y2, score, cls in dets:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {cls}: {score:.2f}"
        cv2.putText(img, label, (x1, max(0, y1-5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"  {label} box=({x1},{y1})-({x2},{y2})")
    
    cv2.imwrite("annotated.jpg", img)
    print("\n✅ Сохранено annotated.jpg")

if __name__ == "__main__":
    main()
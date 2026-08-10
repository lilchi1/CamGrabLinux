# detect_simple.py
import sys
import json
from pathlib import Path

# Все логи и результаты — в папку logs/
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Отключаем matplotlib и другие проблемные импорты
import os
os.environ["ULTRALYTICS_NO_MATPLOTLIB"] = "1"

# Импортируем только то, что нужно
try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("\nПопробуйте понизить версию NumPy:")
    print("  pip uninstall numpy -y")
    print("  pip install numpy<2")
    sys.exit(1)

def main():
    print("=" * 60)
    print("🔍 YOLOv8 ДЕТЕКЦИЯ (упрощенная)")
    print("=" * 60)
    
    model_path = 'yolo26n.pt'
    image_path = 'test.jpg'
    conf_threshold = 0.5
    
    # Проверяем файлы
    if not Path(model_path).exists():
        print(f"❌ Модель не найдена: {model_path}")
        print(f"\nДоступные .pt файлы:")
        for f in Path('.').glob('*.pt'):
            print(f"  - {f.name}")
        return
    
    if not Path(image_path).exists():
        print(f"❌ Изображение не найдено: {image_path}")
        print(f"\nДоступные изображения:")
        for f in Path('.').glob('*.jpg'):
            print(f"  - {f.name}")
        return
    
    print(f"✅ Модель: {model_path}")
    print(f"✅ Изображение: {image_path}")
    
    # Загрузка модели
    print("\n🚀 Загрузка модели...")
    model = YOLO(model_path)
    print("✅ Модель загружена")
    
    # Детекция
    print("\n🔍 Выполнение детекции...")
    results = model(image_path, conf=conf_threshold, iou=0.45)
    
    # Вывод результатов
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ДЕТЕКЦИИ")
    print("=" * 60)
    
    detections = []
    for result in results:
        if result.boxes is None:
            print("ℹ️ Объекты не обнаружены")
            continue
        
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            bbox = [int(x) for x in box.xyxy[0].tolist()]
            
            detections.append({
                'class': label,
                'confidence': conf,
                'bbox': bbox
            })
            
            print(f"✅ {label}: {conf:.3f} [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
    
    print("\n" + "=" * 60)
    print(f"📊 Всего объектов: {len(detections)}")
    
    # Сохраняем в JSON
    detections_path = LOGS_DIR / 'detections.json'
    with open(detections_path, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f"📁 Результаты сохранены в {detections_path}")
    
    # Сохраняем изображение с рамками (используем PIL вместо OpenCV)
    if detections:
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            print("\n🎨 Сохранение изображения с рамками...")
            # Открываем и сразу конвертируем в RGB для совместимости с JPEG
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            # Загружаем шрифт
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = det['class']
                conf = det['confidence']
                
                # Рамка
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                
                # Подпись
                text = f"{label} {conf:.2f}"
                try:
                    text_bbox = draw.textbbox((x1, y1-5), text, font=font)
                except:
                    text_bbox = (x1, y1-20, x1+len(text)*10, y1)
                
                draw.rectangle(text_bbox, fill=(0, 255, 0))
                draw.text((x1, y1-5), text, fill=(0, 0, 0), font=font)
            
            # Сохраняем как JPEG (уже RGB)
            img.save(LOGS_DIR / 'detected_image.jpg', quality=95)
            print(f"✅ Сохранено: {LOGS_DIR / 'detected_image.jpg'}")
            
        except ImportError:
            print("⚠️ PIL не установлен, изображение не сохранено")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения: {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Конвертация YOLO *.pt (Ultralytics) в TensorRT .engine: fp32 / fp16 / int8.

Цепочка (обе ступени официальные):
    .pt  --ultralytics export-->  .onnx   (CPU, torch)
    .onnx  --TensorRT Python API-->  .engine  (GPU)

Точности:
    fp32  — float32, полная точность, без калибровки;
    fp16  — float16, ~2x быстрее на Jetson, без калибровки;
    int8  — int8-квантование, максимальная скорость; ТРЕБУЕТ калибровки
            на реальных изображениях (--calib-dir или --data).

Требования (Jetson, JetPack 6):
    pip install "ultralytics[export]"
    (ставит torch, tensorrt, onnx, onnxslim, pycuda и пр.)

Примеры:
    python3 convert_pt_to_engine.py --pt yolo26n.pt                          # fp16 (по умолчанию)
    python3 convert_pt_to_engine.py --pt yolo26n.pt --precision fp32
    python3 convert_pt_to_engine.py --pt yolo26n.pt --precision fp32 fp16 int8 \
        --calib-dir /data/calib_images                                    # сразу все три
    python3 convert_pt_to_engine.py --pt yolo26n.pt --precision int8 \
        --data /data/coco8.yaml --fraction 0.5

Память (важно для Jetson с общей CPU/GPU RAM):
    Экспорт ONNX выполняется в ОТДЕЛЬНОМ под-процессе, который завершается и
    освобождает torch до начала сборки engine. Это нужно, т.к. torch держит
    ~1.5-2 ГБ, а на сборку TensorRT их не остаётся (NvMapMemAlloc error 12,
    "terminate called without an active exception"). Перед сборкой проверьте
    свободную память: free -h. Сборка занимает несколько минут — не прерывайте.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

try:
    import tensorrt as trt
except ImportError:
    sys.exit("tensorrt (python) не установлен — выполните: pip install tensorrt")


# ─────────────────────────── int8: калибровка ────────────────────────────


def collect_calib_images(args):
    """Список изображений для int8-калибровки из --calib-dir или --data yaml."""
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    imgs = []
    if args.calib_dir:
        if not os.path.isdir(args.calib_dir):
            sys.exit(f"Папка калибровки не найдена: {args.calib_dir}")
        imgs = [os.path.join(args.calib_dir, f) for f in os.listdir(args.calib_dir)
                if f.lower().endswith(exts)]
    elif args.data:
        import yaml
        with open(args.data) as f:
            data = yaml.safe_load(f)
        # Относительный path в yaml отсчитывается от папки самого yaml
        # (конвенция Ultralytics), а не от текущего каталога запуска.
        yaml_dir = os.path.dirname(os.path.abspath(args.data))
        root = data.get("path")
        if root and not os.path.isabs(root):
            root = os.path.join(yaml_dir, root)
        elif not root:
            root = yaml_dir
        for split in ("val", "train"):
            d = data.get(split)
            if not d:
                continue
            entries = d if isinstance(d, list) else [d]
            for p in entries:
                full = p if os.path.isabs(p) else os.path.join(root, p)
                if os.path.isdir(full):
                    imgs += [os.path.join(full, x) for x in os.listdir(full)
                             if x.lower().endswith(exts)]
                elif os.path.isfile(full) and full.lower().endswith(exts):
                    imgs.append(full)
    else:
        sys.exit("int8 требует калибровочных данных: --calib-dir <папка с фото> "
                 "или --data <data.yaml>")
    imgs = sorted(set(imgs))
    if args.fraction is not None:
        imgs = imgs[: max(1, round(len(imgs) * args.fraction))]
    if not imgs:
        sys.exit("Не найдено изображений для калибровки")
    return imgs


def load_and_letterbox(path, imgsz):
    """Читает изображение, letterbox до imgsz, возвращает CHW float32 [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Не удалось прочитать изображение: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    r = imgsz / max(h, w)
    new_w, new_h = round(w * r), round(h * r)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top = (imgsz - new_h) // 2
    bottom = imgsz - new_h - top
    left = (imgsz - new_w) // 2
    right = imgsz - new_w - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """Энтропийный калибратор TensorRT для int8.

    Подаёт letterbox-изображения [0,1] float32 NCHW, как при инференсе.
    Собранный кэш сохраняется рядом с engine (*.cache) и переиспользуется.
    """
    def __init__(self, image_paths, batch_size, imgsz, cache_path):
        import pycuda.driver as cuda
        cuda.init()
        # Активный CUDA-контекст обязателен для mem_alloc в get_batch.
        # Обычно его уже создал pycuda.autoinit в build_trt_engine (ДО
        # createInferBuilder); здесь создаём только если его нет.
        self._ctx = None
        if cuda.Context.get_current() is None:
            from pycuda.tools import make_default_context
            self._ctx = make_default_context()
        super().__init__()
        self._cuda = cuda
        self._paths = image_paths
        self._batch_size = batch_size
        self._imgsz = imgsz
        self._cache_path = cache_path
        self._idx = 0
        self._device = None
        self.cache_path = cache_path

    def get_batch_size(self):
        return self._batch_size

    def get_batch(self, names):
        if self._idx >= len(self._paths):
            return None
        batch = []
        for _ in range(self._batch_size):
            p = self._paths[self._idx % len(self._paths)]
            self._idx += 1
            batch.append(load_and_letterbox(p, self._imgsz))
        # ascontiguousarray: после transpose(2,0,1) массив не C-смежный,
        # memcpy_htod требует непрерывный буфер.
        arr = np.ascontiguousarray(np.stack(batch), dtype=np.float32)
        self._device = self._cuda.mem_alloc(arr.nbytes)
        self._cuda.memcpy_htod(self._device, arr)
        return [int(self._device)]

    def read_calibration_cache(self):
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self._cache_path, "wb") as f:
            f.write(cache)


# ─────────────────────────── сборка engine ───────────────────────────────


def export_onnx(args):
    """.pt -> ONNX (ultralytics, CPU). ONNX экспортируется один раз.

    Загружает torch/ultralytics, поэтому должен вызываться в отдельном процессе
    (--export-only), чтобы освободить память до сборки TensorRT.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics не установлен — выполните: pip install 'ultralytics[export]'")

    try:
        model = YOLO(args.pt)
    except Exception as e:
        sys.exit(f"Не удалось загрузить {args.pt}: {e}")

    print("\n[1/2] Экспорт ONNX (ultralytics, CPU)...")
    try:
        onnx_path = model.export(
            format="onnx",
            imgsz=args.imgsz,
            batch=args.batch,
            device="cpu",
            opset=13,
            end2end=False,
            nms=False,
            simplify=True,
        )
    except Exception as e:
        sys.exit(f"Экспорт ONNX завершился ошибкой: {e}")
    onnx_path = str(onnx_path)
    print(f"    ONNX: {onnx_path}")
    return onnx_path


def build_engine(args, onnx_path, precision):
    """.onnx -> .engine (TensorRT, GPU) для одной точности."""
    print(f"\n=== {os.path.basename(args.pt)} -> TensorRT ({precision}) ===")
    print(f"    imgsz={args.imgsz}, batch={args.batch}, workspace={args.workspace}GB")

    calib_images = collect_calib_images(args) if precision == "int8" else []
    if precision == "int8":
        print(f"    калибровочных изображений: {len(calib_images)}")

    engine_path = args.engine or default_engine_path(args.pt, precision)
    print(f"[2/2] Сборка engine TensorRT ({precision}, GPU)...")
    build_trt_engine(onnx_path, engine_path, precision, args.imgsz,
                     args.batch, args.workspace, calib_images)

    print_engine_io(engine_path)
    return engine_path


def build_trt_engine(onnx_path, engine_path, precision, imgsz, batch,
                     workspace_gb, calib_images):
    """Парсит ONNX и собирает .engine; int8 — с энтропийной калибровкой."""
    logger = trt.Logger(trt.Logger.WARNING)
    # Единый CUDA-контекст на весь билд, строго ДО createInferBuilder: если
    # контекст появится позже (например, из pycuda в калибраторе), TensorRT
    # теряет свои ресурсы при смене контекста и все тактики падают с
    # "Cask convolution execution" -> "Could not find any implementation".
    import pycuda.autoinit  # noqa: F401
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("ONNX parse error:", parser.get_error(i))
            sys.exit("Не удалось разобрать ONNX")
    print(f"    сеть: входов={network.num_inputs}, выходов={network.num_outputs}")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"      вход '{t.name}' shape={list(t.shape)} dtype={t.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024 ** 3))
    if precision in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        cache_path = os.path.splitext(engine_path)[0] + ".cache"
        calibrator = ImageCalibrator(calib_images, batch, imgsz, cache_path)
        config.int8_calibrator = calibrator
        print(f"    int8: калибровка {len(calib_images)} изображений -> {cache_path}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        sys.exit("Сборка engine завершилась неудачей (build_serialized_network=None)")

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    size_mb = os.path.getsize(engine_path) / 1024 ** 2
    print(f"    engine сохранён: {engine_path} ({size_mb:.1f} MB)")


def print_engine_io(engine_path):
    """Печать входных/выходных тензоров .engine через TRT Python API."""
    try:
        import tensorrt as trt
    except ImportError:
        print("tensorrt (python) не установлен — печать I/O engine пропущена.")
        return
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    print("=== TensorRT engine I/O ===")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        print(f"  {mode} '{name}' shape={list(shape)} dtype={dtype}")


def default_engine_path(pt, precision):
    stem = os.path.splitext(os.path.basename(pt))[0]
    return os.path.join(os.path.dirname(pt) or ".",
                        f"{stem}_{precision}.engine")


def default_onnx_path(pt):
    """ONNX сохраняется ultralytics рядом с .pt: <имя>.onnx."""
    return os.path.splitext(pt)[0] + ".onnx"


def main():
    ap = argparse.ArgumentParser(
        description="YOLO *.pt (Ultralytics) -> TensorRT .engine (fp32/fp16/int8)")
    ap.add_argument("--pt", required=True, help="исходная модель .pt")
    ap.add_argument("--engine", default=None,
                    help="выходной .engine (только для одной точности; "
                         "по умолчанию <имя>_<точность>.engine)")
    ap.add_argument("--precision", nargs="+", default=["fp16"],
                    help="точность: fp32/fp16/int8; можно несколько (по умолчанию fp16)")
    ap.add_argument("--imgsz", type=int, default=640, help="размер входа (640)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workspace", type=float, default=0.5,
                    help="лимит памяти TensorRT, ГБ (по умолчанию 0.5; на Jetson "
                         "GPU-память = общая RAM, большие значения дают OOM)")
    ap.add_argument("--data", default=None,
                    help="data.yaml для int8-калибровки (val/train = фото)")
    ap.add_argument("--calib-dir", default=None,
                    help="папка с фото для int8-калибровки")
    ap.add_argument("--fraction", type=float, default=None,
                    help="какая доля набора идёт на калибровку int8 (0.5 = половина)")
    ap.add_argument("--keep-onnx", action="store_true",
                    help="не удалять промежуточный .onnx")
    ap.add_argument("--export-only", action="store_true",
                    help="только экспорт .pt -> .onnx и выход (внутреннее)")
    ap.add_argument("--build-only", action="store_true",
                    help="пропустить экспорт, если .onnx уже существует")
    args = ap.parse_args()

    if not os.path.isfile(args.pt):
        sys.exit(f"Файл не найден: {args.pt}")

    precisions = [p.lower() for token in args.precision for p in token.split(",")]
    for p in precisions:
        if p not in ("fp32", "fp16", "int8"):
            sys.exit(f"Неизвестная точность: {p} (ожидается fp32/fp16/int8)")

    onnx_path = default_onnx_path(args.pt)

    if args.export_only:
        export_onnx(args)
        return

    if args.build_only or os.path.exists(onnx_path):
        if not os.path.exists(onnx_path):
            sys.exit("ONNX не найден. Запустите без --build-only.")
        print(f"ONNX уже существует: {onnx_path} (пропуск экспорта)")
    else:
        # Экспорт в отдельном процессе: он загружает torch (~1.5-2 ГБ) и после
        # завершения освободит память, чтобы её получила сборка TensorRT.
        print("Экспорт ONNX в отдельном процессе (torch освободит память)...")
        cmd = [sys.executable, os.path.abspath(__file__), "--export-only",
               "--pt", args.pt,
               "--imgsz", str(args.imgsz),
               "--batch", str(args.batch)]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit(f"Экспорт ONNX завершился с кодом {rc}.")
        if not os.path.exists(onnx_path):
            sys.exit(f"Экспорт завершён, но {onnx_path} не найден.")
    if "int8" in precisions and not args.calib_dir and not args.data:
        sys.exit("int8 требует калибровочных данных: --calib-dir <папка с фото> "
                 "или --data <data.yaml>")
    if len(precisions) > 1 and args.engine:
        sys.exit("--engine задаёт один файл — укажите одну точность "
                 "или уберите --engine (пути выведутся сами)")

    engines = [build_engine(args, onnx_path, p) for p in precisions]

    if os.path.exists(onnx_path) and not args.keep_onnx:
        os.unlink(onnx_path)
        print("Удалён промежуточный ONNX:", onnx_path)

    print("\nГотовые engine:")
    for e in engines:
        print(f"  {e}")
    print("\nЗапуск в проекте:")
    for e in engines:
        print(f"  ./Build/rtsp_decoder -d cuda -m {e} "
              f"-l yolo/coco.names -w 1200 -H 800")


if __name__ == "__main__":
    main()

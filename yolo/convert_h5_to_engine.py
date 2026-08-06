#!/usr/bin/env python3
"""Конвертация Keras YOLOv2-модели (yolo_model_complete.h5) в TensorRT .engine.

Цепочка:  .h5 (Keras)  --tf2onnx-->  .onnx  --trtexec-->  .engine

Особенности модели (извлечено из архива h5):
  - вход  : NHWC [1, 608, 608, 3] (F32, 0..1, т.е. нормализация /255)
  - выход : NHWC [1, 19, 19, 425], 425 = 5 якорей * (5 + 80 классов COCO)
  - есть кастомный слой Lambda space_to_depth (passthrough YOLOv2)
  - декод YOLOv2: sigmoid(tx/ty/to), exp(tw/th), сетка 19x19, stride = 32

Скрипт:
  1. Загружает модель (с custom_objects для Lambda).
  2. Оборачивает её: вход NCHW [1,3,608,608], выход NCHW [1,425,19,19]
     (этот формат ожидает C++-постпроцессор проекта).
  3. Экспортирует в ONNX (tf2onnx, вход как NCHW).
  4. Собирает .engine через trtexec (FP16 — быстро на Jetson).
  5. Опционально проверяет ONNX и печатает I/O-тензоры engine (TRT Python API).

Зависимости (установить на машине с интернетом):
    pip install tensorflow tf2onnx numpy opencv-python-headless onnxruntime

Пример:
    python3 convert_h5_to_engine.py
    python3 convert_h5_to_engine.py --h5 my_model.h5 --engine my_model.engine
    python3 convert_h5_to_engine.py --anchors 20,24,60,66,107,175,252,113,313,293
"""
import argparse
import os
import subprocess
import sys

MODEL_PATH = "yolo_model_complete.h5"
DEFAULT_TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

# Якоря по умолчанию (YOLOv2 COCO, YAD2K): нормированные на сетку 13 * stride 32
# = пиксельные якоря для входа 608 (и 416). Пары w,h в пикселях входа.
DEFAULT_ANCHORS_PX = [18.32736, 21.67632,
                      59.98272, 66.00096,
                      106.82976, 175.17888,
                      252.25024, 112.88896,
                      312.65664, 293.38496]


def space_to_depth_x2(x):
    """Lambda passthrough YOLOv2: сжатие признаков 2x2 -> глубина*4."""
    import tensorflow as tf
    return tf.nn.space_to_depth(x, block_size=2)


def build_wrapped_model(h5_path):
    """Загружает h5 и возвращает модель с входом/выходом NCHW [1,3,H,W]/[1,C,H,W]."""
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    model = load_model(h5_path, compile=False,
                       custom_objects={"space_to_depth_x2": space_to_depth_x2})
    print("Загружена модель:", h5_path)
    model.summary()

    in_shape = model.inputs[0].shape  # (None, H, W, 3)
    h, w = int(in_shape[1]), int(in_shape[2])
    print(f"Вход модели NHWC {in_shape} -> {w}x{h}")

    inp = tf.keras.layers.Input(shape=(h, w, 3), name="input")
    x = model(inp)                       # NHWC [1, h/32, w/32, 5*(5+nc)]
    out = tf.transpose(x, [0, 3, 1, 2])  # NCHW [1, C, grid, grid]

    wrapped = tf.keras.Model(inp, out, name="yolo_v2_nchw")
    print("Обёртка: вход NCHW, выход NCHW:", wrapped.outputs[0].shape)
    return wrapped, w


def export_onnx(model, onnx_path):
    """Keras -> ONNX (tf2onnx). Вход графа конвертируется в NCHW внутри."""
    import tf2onnx

    model_proto, _ = tf2onnx.convert.from_keras(
        model, opset=13, inputs_as_nchw=["input"], output_path=onnx_path)
    print("ONNX сохранён:", onnx_path)


def build_engine(onnx_path, engine_path, trtexec, fp16, extra):
    cmd = [trtexec, "--onnx=" + onnx_path, "--saveEngine=" + engine_path]
    if fp16:
        cmd.append("--fp16")
    if extra:
        cmd += extra
    print("Запуск:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"trtexec завершился с кодом {r.returncode}")
    print("Engine сохранён:", engine_path)


def check_onnx(onnx_path, grid, num_classes):
    """Проверка входной/выходной формы ONNX через onnxruntime (если установлен)."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime не установлен — проверка ONNX пропущена.")
        return
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print("ONNX вход:", inp.name, inp.shape, inp.type)
    print("ONNX выход:", out.name, out.shape, out.type)
    expected = [1, 5 * (5 + num_classes), grid, grid]
    print("Ожидаемый выход:", expected,
          "(1, 5*(5+classes), grid, grid) — NCHW, каналы-major")


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


def main():
    ap = argparse.ArgumentParser(description="Keras YOLOv2 h5 -> TensorRT engine")
    ap.add_argument("--h5", default=MODEL_PATH)
    ap.add_argument("--onnx", default="yolo_v2.onnx")
    ap.add_argument("--engine", default="yolo_v2.engine")
    ap.add_argument("--trtexec", default=DEFAULT_TRTEXEC)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--skip-export", action="store_true",
                    help="использовать существующий ONNX, только собрать engine")
    ap.add_argument("--trtexec-args", default="", help="доп. аргументы trtexec (строка)")
    ap.add_argument("--anchors", default=",".join(map(str, DEFAULT_ANCHORS_PX)),
                    help="пиксельные якоря YOLOv2, пары w,h (по умолчанию COCO YAD2K)")
    ap.add_argument("--grid", type=int, default=19, help="сетка выхода (608/32=19)")
    ap.add_argument("--classes", type=int, default=80, help="число классов (425 = 5*(5+80))")
    args = ap.parse_args()

    anchors = [float(v) for v in args.anchors.split(",")]
    if len(anchors) % 2 != 0 or not anchors:
        sys.exit("--anchors должно быть чётным числом значений (w,h,...)")
    input_size = args.grid * 32

    if not args.skip_export:
        model, _ = build_wrapped_model(args.h5)
        export_onnx(model, args.onnx)
    else:
        print("Пропуск экспорта ONNX (--skip-export).")

    check_onnx(args.onnx, args.grid, args.classes)

    if os.path.exists(args.trtexec):
        build_engine(args.onnx, args.engine, args.trtexec, args.fp16,
                     args.trtexec_args.split())
        print_engine_io(args.engine)
    else:
        print("trtexec не найден:", args.trtexec,
              "— соберите engine вручную или укажите --trtexec.")

    print("\nЗапуск в проекте:")
    print(f"  ./rtsp_decoder --model {args.engine} --labels coco.names --yolov2 \\")
    print(f"      --v2-grid {args.grid} --v2-anchors \"{','.join(map(str, anchors))}\"")


if __name__ == "__main__":
    main()

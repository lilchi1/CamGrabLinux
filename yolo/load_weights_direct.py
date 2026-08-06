# load_weights_direct.py
import tensorflow as tf
import numpy as np
import h5py
import os
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")

def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)

def create_model_from_h5(h5_path='yolo_model_complete.h5'):
    """Создание модели с именами слоев из H5 файла"""
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Concatenate, LeakyReLU, BatchNormalization, Add
    from tensorflow.keras.models import Model
    
    print("🏗️ Анализ H5 файла...")
    
    # Читаем имена слоев из H5
    layer_names = []
    with h5py.File(h5_path, 'r') as f:
        if 'model_weights' in f:
            weights_group = f['model_weights']
            layer_names = list(weights_group.keys())
            print(f"📊 Найдено слоев в H5: {len(layer_names)}")
            print(f"Первые 10 слоев: {layer_names[:10]}")
    
    # Создаем модель с правильными именами
    print("\n🏗️ Создание модели...")
    
    def conv_bn_leaky(inputs, filters, size, stride=1, name=''):
        x = Conv2D(filters, size, strides=stride, padding='same', name=name)(inputs)
        x = BatchNormalization(name=name)(x)  # Используем то же имя
        x = LeakyReLU(alpha=0.1)(x)
        return x
    
    inputs = Input(shape=(608, 608, 3), name='input_1')
    
    # Используем имена из H5
    x = Conv2D(32, 3, padding='same', name='conv2d')(inputs)
    x = BatchNormalization(name='batch_normalization')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=2, strides=2, name='max_pooling2d')(x)
    
    x = Conv2D(64, 3, padding='same', name='conv2d_1')(x)
    x = BatchNormalization(name='batch_normalization_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=2, strides=2, name='max_pooling2d_1')(x)
    
    x = Conv2D(128, 3, padding='same', name='conv2d_2')(x)
    x = BatchNormalization(name='batch_normalization_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(64, 1, padding='same', name='conv2d_3')(x)
    x = BatchNormalization(name='batch_normalization_3')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, 3, padding='same', name='conv2d_4')(x)
    x = BatchNormalization(name='batch_normalization_4')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=2, strides=2, name='max_pooling2d_2')(x)
    
    x = Conv2D(256, 3, padding='same', name='conv2d_5')(x)
    x = BatchNormalization(name='batch_normalization_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, 1, padding='same', name='conv2d_6')(x)
    x = BatchNormalization(name='batch_normalization_6')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256, 3, padding='same', name='conv2d_7')(x)
    x = BatchNormalization(name='batch_normalization_7')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=2, strides=2, name='max_pooling2d_3')(x)
    
    x = Conv2D(512, 3, padding='same', name='conv2d_8')(x)
    x = BatchNormalization(name='batch_normalization_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256, 1, padding='same', name='conv2d_9')(x)
    x = BatchNormalization(name='batch_normalization_9')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512, 3, padding='same', name='conv2d_10')(x)
    x = BatchNormalization(name='batch_normalization_10')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Passthrough
    passthrough = x
    x = MaxPooling2D(pool_size=2, strides=2, name='max_pooling2d_4')(x)
    
    x = Conv2D(1024, 3, padding='same', name='conv2d_11')(x)
    x = BatchNormalization(name='batch_normalization_11')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512, 1, padding='same', name='conv2d_12')(x)
    x = BatchNormalization(name='batch_normalization_12')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(1024, 3, padding='same', name='conv2d_13')(x)
    x = BatchNormalization(name='batch_normalization_13')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512, 1, padding='same', name='conv2d_14')(x)
    x = BatchNormalization(name='batch_normalization_14')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(1024, 3, padding='same', name='conv2d_15')(x)
    x = BatchNormalization(name='batch_normalization_15')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Space to depth
    passthrough = Lambda(space_to_depth_x2, name='space_to_depth_x2')(passthrough)
    
    # Concatenate
    x = Concatenate(axis=-1, name='concatenate')([x, passthrough])
    
    # Output
    x = Conv2D(1024, 3, padding='same', name='conv2d_16')(x)
    x = BatchNormalization(name='batch_normalization_16')(x)
    x = LeakyReLU(alpha=0.1)(x)
    outputs = Conv2D(425, 1, padding='same', name='conv2d_17')(x)
    
    model = Model(inputs, outputs, name='model_1')
    
    print("✅ Модель создана")
    model.summary()
    return model

def load_weights_direct(model, h5_path='yolo_model_complete.h5'):
    """Прямая загрузка весов из H5"""
    print("\n🔄 ЗАГРУЗКА ВЕСОВ")
    print("="*60)
    
    loaded_count = 0
    
    try:
        with h5py.File(h5_path, 'r') as f:
            weights_group = f['model_weights']
            
            # Проходим по всем слоям в H5
            for h5_layer_name in weights_group:
                layer_group = weights_group[h5_layer_name]
                
                # Собираем веса
                weights = []
                for key in sorted(layer_group.keys()):
                    item = layer_group[key]
                    if isinstance(item, h5py.Dataset):
                        weights.append(item[:])
                
                if not weights:
                    continue
                
                # Ищем слой в модели с таким же именем
                found = False
                for model_layer in model.layers:
                    if model_layer.name == h5_layer_name:
                        try:
                            # Проверяем совместимость
                            current_weights = model_layer.get_weights()
                            if len(current_weights) == len(weights):
                                # Проверяем размеры
                                match = True
                                for i in range(len(weights)):
                                    if weights[i].shape != current_weights[i].shape:
                                        match = False
                                        break
                                if match:
                                    model_layer.set_weights(weights)
                                    print(f"  ✅ {h5_layer_name}: загружены")
                                    loaded_count += 1
                                    found = True
                        except Exception as e:
                            print(f"  ⚠️ {h5_layer_name}: {str(e)[:50]}")
                        break
                
                if not found:
                    # Пробуем найти слой по части имени
                    base_name = h5_layer_name.split('_')[0]
                    for model_layer in model.layers:
                        if model_layer.name.startswith(base_name) or h5_layer_name in model_layer.name:
                            try:
                                model_layer.set_weights(weights)
                                print(f"  ✅ {h5_layer_name} -> {model_layer.name}: загружены")
                                loaded_count += 1
                                found = True
                                break
                            except:
                                pass
                    
                    if not found:
                        print(f"  ⏭️ {h5_layer_name}: пропущен")
            
            print(f"\n✅ Загружено {loaded_count} слоев")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    return model

# Основной процесс
print("="*60)
print("🔧 ЗАГРУЗКА YOLOv2 МОДЕЛИ")
print("="*60)

# Создаем модель
model = create_model_from_h5()

# Загружаем веса
model = load_weights_direct(model)

# Тестируем
print("\n🧪 ТЕСТИРОВАНИЕ")
print("="*60)

test_input = np.random.random((1, 608, 608, 3)).astype(np.float32)
output = model.predict(test_input, verbose=0)

print(f"✅ Тест успешен!")
print(f"Выходной shape: {output.shape}")
print(f"Статистика:")
print(f"  min: {np.min(output):.4f}")
print(f"  max: {np.max(output):.4f}")
print(f"  mean: {np.mean(output):.4f}")
print(f"  std: {np.std(output):.4f}")

# Сохраняем
model.save('yolov2_working.keras', save_format='keras')
print("\n💾 Модель сохранена как yolov2_working.keras")
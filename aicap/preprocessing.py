"""Общий слой загрузки и валидации изображений для всех детекторов.

Единая точка, где файл открывается и проверяется, чтобы fft_detector,
exif_detector и cnn_detector не дублировали эту логику каждый по-своему,
и чтобы битый файл не ронял весь прогон на датасете в experiments/.
"""

import os

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CNN_INPUT_SIZE = 224
CNN_MEAN = (0.485, 0.456, 0.406)
CNN_STD = (0.229, 0.224, 0.225)


class InvalidImageError(Exception):
    """Файл не удалось использовать как изображение."""


def validate_image(path):
    """(True, None), если файл валиден, иначе (False, причина отказа)."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"неподдерживаемое расширение: {ext or '(нет)'}"

    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        return False, f"файл повреждён или не является изображением: {exc}"

    return True, None


def load_image(path):
    """Открывает изображение как PIL.Image в режиме RGB.

    Бросает InvalidImageError на невалидном файле — так батч-обработка в
    experiments/ может поймать конкретно это исключение и пропустить файл
    с пометкой в отчёте, не прерывая весь прогон.
    """
    is_valid, reason = validate_image(path)
    if not is_valid:
        raise InvalidImageError(f"{path}: {reason}")

    return Image.open(path).convert("RGB")


def to_grayscale_array(image):
    """PIL.Image -> np.ndarray float32 в градациях серого. Используется fft_detector."""
    return np.asarray(image.convert("L"), dtype=np.float32)


def to_cnn_array(image):
    """PIL.Image -> np.ndarray (H, W, 3) float32, resize 224x224 + ImageNet-нормализация,
    как ожидает CLIP-энкодер в cnn_detector.py."""
    resized = image.convert("RGB").resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE), Image.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array(CNN_MEAN, dtype=np.float32)
    std = np.array(CNN_STD, dtype=np.float32)
    return (array - mean) / std


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Проверить файл и посмотреть, что из него получится")
    parser.add_argument("image", help="Путь к изображению")
    args = parser.parse_args()

    is_valid, reason = validate_image(args.image)
    if not is_valid:
        print(f"НЕВАЛИДНО: {reason}")
        raise SystemExit(1)

    image = load_image(args.image)
    gray = to_grayscale_array(image)
    cnn_ready = to_cnn_array(image)

    print(f"Валидно. Формат: {image.format or os.path.splitext(args.image)[1]}, размер: {image.size}")
    print(f"Grayscale-массив (для FFT): {gray.shape}, dtype={gray.dtype}")
    print(f"CNN-массив (для CLIP): {cnn_ready.shape}, dtype={cnn_ready.dtype}, "
          f"диапазон=[{cnn_ready.min():.2f}, {cnn_ready.max():.2f}]")

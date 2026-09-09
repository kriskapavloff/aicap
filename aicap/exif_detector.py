"""EXIF-детектор: смотрит на метаданные изображения — есть ли признаки реальной
съёмки и нет ли явного маркера генератора в поле Software.

В отличие от FFT и CLIP, это не числовой score. Метаданные слишком легко
потерять (любая пересохранка/скриншот их стирает) или подделать вручную,
поэтому в агрегацию этот сигнал не идёт — он попадает в отчёт как
качественное доказательство. Это осознанное архитектурное решение из
диплома, сохранено намеренно.
"""

import piexif

from . import preprocessing

CAMERA_FIELDS = {
    "Camera Make": ("0th", piexif.ImageIFD.Make),
    "Camera Model": ("0th", piexif.ImageIFD.Model),
    "Lens Model": ("Exif", piexif.ExifIFD.LensModel),
    "MakerNote": ("Exif", piexif.ExifIFD.MakerNote),
    "Exposure Time": ("Exif", piexif.ExifIFD.ExposureTime),
    "F-Number": ("Exif", piexif.ExifIFD.FNumber),
    "ISO": ("Exif", piexif.ExifIFD.ISOSpeedRatings),
    "Original Date/Time": ("Exif", piexif.ExifIFD.DateTimeOriginal),
}

KNOWN_GENERATOR_MARKERS = [
    "midjourney", "stable diffusion", "dall-e", "dalle",
    "adobe firefly", "firefly", "imagen", "leonardo", "runway",
]


def read_exif(path):
    """Возвращает словарь EXIF или None, если его нет / формат его не поддерживает
    (например, у большинства PNG нет EXIF-сегмента вообще — piexif рассчитан на JPEG/TIFF)."""
    try:
        return piexif.load(str(path))
    except Exception:
        return None


def check_camera_fields(exif_dict):
    if exif_dict is None:
        return [], list(CAMERA_FIELDS.keys())

    present, absent = [], []
    for name, (ifd, tag) in CAMERA_FIELDS.items():
        value = exif_dict.get(ifd, {}).get(tag)
        (present if value else absent).append(name)

    return present, absent


def check_generator_marker(exif_dict):
    if exif_dict is None:
        return False, None, None

    software = exif_dict.get("0th", {}).get(piexif.ImageIFD.Software)
    if not software:
        return False, None, None

    software_str = software.decode("utf-8", errors="ignore") if isinstance(software, bytes) else str(software)
    lowered = software_str.lower()

    for marker in KNOWN_GENERATOR_MARKERS:
        if marker in lowered:
            return True, marker, software_str

    return False, None, software_str


def analyze(path):
    is_valid, reason = preprocessing.validate_image(path)
    if not is_valid:
        raise preprocessing.InvalidImageError(f"{path}: {reason}")

    exif_dict = read_exif(path)
    present, absent = check_camera_fields(exif_dict)
    marker_found, marker_name, software_value = check_generator_marker(exif_dict)

    return {
        "has_exif_container": exif_dict is not None,
        "present_camera_fields": present,
        "absent_camera_fields": absent,
        "looks_like_real_camera": len(present) > 0,
        "generator_marker_found": marker_found,
        "generator_marker_name": marker_name,
        "software_field": software_value,
    }


def format_report(result):
    lines = []
    if not result["has_exif_container"]:
        lines.append("EXIF отсутствует полностью (типично для PNG и для многих экспортов из ИИ-генераторов).")
    else:
        if result["present_camera_fields"]:
            lines.append("Найдены поля реальной съёмки: " + ", ".join(result["present_camera_fields"]))
        else:
            lines.append("EXIF-контейнер есть, но полей реальной съёмки не найдено.")
        lines.append("Отсутствуют: " + ", ".join(result["absent_camera_fields"]))

    if result["generator_marker_found"]:
        lines.append(f"НАЙДЕН МАРКЕР ГЕНЕРАТОРА в поле Software: «{result['generator_marker_name']}» "
                      f"(значение поля: {result['software_field']})")
    elif result["software_field"]:
        lines.append(f"Поле Software присутствует, но без известных маркеров: «{result['software_field']}»")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Проверить EXIF-метаданные изображения")
    parser.add_argument("image", help="Путь к изображению")
    args = parser.parse_args()

    result = analyze(args.image)
    print(format_report(result))

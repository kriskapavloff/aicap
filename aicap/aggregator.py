"""Агрегатор: объединяет числовые оценки FFT и CNN-детекторов в единый score
и метку риска, и собирает отчёт по картинке со всеми детекторами разом.

score = FFT × 0.3 + CNN × 0.7

EXIF в этот score не входит — см. exif_detector.py: это качественное
доказательство для отчёта, а не числовой сигнал (метаданные слишком легко
подделать или стереть).

Веса и пороги ниже — из диплома, где они были обоснованы интервью с
представителем приёмной комиссии. Это ЭМПИРИЧЕСКИЕ значения, ещё не
проверенные на данных — задача experiments/: подтвердить или пересчитать
и честно описать в README, если они не подтвердятся.
"""

from . import cnn_detector, exif_detector, fft_detector

FFT_WEIGHT = 0.3
CNN_WEIGHT = 0.7

RISK_THRESHOLD_MEDIUM = 0.5
RISK_THRESHOLD_HIGH = 0.7


def aggregate(fft_score, cnn_score, fft_weight=FFT_WEIGHT, cnn_weight=CNN_WEIGHT):
    """Взвешенная сумма FFT и CNN. Если cnn_score is None (классификатор ещё
    не обучен), возвращает None, а не подделывает число за счёт одного FFT —
    это исказило бы картину, будто система работает целиком."""
    if cnn_score is None:
        return None
    return fft_weight * fft_score + cnn_weight * cnn_score


def risk_label(score):
    if score is None:
        return "неизвестно (недостаточно данных)"
    if score < RISK_THRESHOLD_MEDIUM:
        return "низкий"
    if score < RISK_THRESHOLD_HIGH:
        return "средний"
    return "высокий"


def analyze_image(path, run_cnn=True, cnn_classifier_path=None):
    """Прогоняет FFT, EXIF и (опционально) CNN-детекторы и собирает единый отчёт.

    run_cnn=False позволяет протестировать пайплайн без CLIP — полезно, пока
    веса CLIP не скачаны или соединение нестабильно.
    """
    fft_result = fft_detector.analyze(path)
    exif_result = exif_detector.analyze(path)

    cnn_result = None
    if run_cnn:
        kwargs = {"classifier_path": cnn_classifier_path} if cnn_classifier_path else {}
        cnn_result = cnn_detector.analyze(path, **kwargs)

    cnn_score = cnn_result["score"] if cnn_result else None
    combined = aggregate(fft_result["score"], cnn_score)

    return {
        "fft": fft_result,
        "exif": exif_result,
        "cnn": cnn_result,
        "aggregate_score": combined,
        "risk_label": risk_label(combined),
    }


def format_report(result):
    lines = [f"FFT: score={result['fft']['score']:.3f} (rmse={result['fft']['rmse']:.3f})"]

    if result["cnn"] is None:
        lines.append("CNN (CLIP): не запускался в этом прогоне (--skip-cnn)")
    elif not result["cnn"]["classifier_trained"]:
        lines.append("CNN (CLIP): признаки извлечены, но классификатор ещё не обучен")
    else:
        lines.append(f"CNN (CLIP): score={result['cnn']['score']:.3f}")

    if result["aggregate_score"] is None:
        lines.append("Итоговый score: недоступен (нужен обученный CNN-классификатор)")
    else:
        lines.append(f"Итоговый score: {result['aggregate_score']:.3f}")
    lines.append(f"Риск: {result['risk_label']}")

    lines.append("")
    lines.append("EXIF (доказательная база, не входит в score):")
    lines.append(exif_detector.format_report(result["exif"]))

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Полный прогон всех детекторов по картинке")
    parser.add_argument("image", help="Путь к изображению")
    parser.add_argument("--skip-cnn", action="store_true",
                         help="Не запускать CLIP (например, если веса ещё не скачаны)")
    args = parser.parse_args()

    result = analyze_image(args.image, run_cnn=not args.skip_cnn)
    print(format_report(result))

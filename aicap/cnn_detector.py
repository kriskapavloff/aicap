"""CNN-детектор: подход UnivFD (Ojha et al., "Towards Universal Fake Image
Detectors that Generalize Across Generative Models", CVPR 2023).

Замороженный CLIP ViT-L/14 как экстрактор признаков + обучаемый линейный
классификатор (логистическая регрессия) поверх. CLIP обучен на сотнях
миллионов разнородных пар изображение-текст, поэтому лучше обобщается на
нефотографический контент — иллюстрации, арт, дизайн. Это важно, потому что
предметная область — художественные портфолио, а не фотографии, на которых
калибруется большинство детекторов ИИ-изображений.

ВАЖНО (честность, см. README): классификатор здесь НЕ обучен. Готовых весов
линейного классификатора от авторов UnivFD под ViT-L/14-openai в открытом
доступе нет, а обучать свой без реального датасета — нечестно. Обучение
происходит в experiments/run_dataset.py, когда датасет собран. До этого
analyze() честно возвращает classifier_trained=False и score=None, а не
подделывает число.
"""

import os
import pickle

import numpy as np
import open_clip
import torch

from . import preprocessing

_MODEL_NAME = "ViT-L-14"
_PRETRAINED = "openai"
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "open_clip")
_DEFAULT_CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "cnn_classifier.pkl")

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_clip_preprocess = None


def _get_model():
    """Ленивая загрузка CLIP — тяжёлая операция (веса качаются один раз и кэшируются
    на диске), делаем один раз за процесс, а не при каждом вызове analyze()."""
    global _model, _clip_preprocess
    if _model is None:
        _model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED, cache_dir=_CACHE_DIR,
        )
        _model.eval().to(_device)
    return _model, _clip_preprocess


def extract_features(image_path):
    """Путь к изображению -> L2-нормализованный вектор признаков CLIP
    (768 чисел для ViT-L/14), как numpy-массив."""
    model, clip_preprocess = _get_model()
    image = preprocessing.load_image(image_path)
    tensor = clip_preprocess(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze(0).cpu().numpy()


def train_classifier(features, labels):
    """features: (N, D) признаки CLIP, labels: (N,) 0=реальное/1=ИИ.

    Логистическая регрессия поверх замороженных признаков — так делали и
    авторы UnivFD: простой линейный классификатор, без переобучения CLIP.
    """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000)
    clf.fit(features, labels)
    return clf


def save_classifier(clf, path=_DEFAULT_CLASSIFIER_PATH):
    with open(path, "wb") as f:
        pickle.dump(clf, f)


def load_classifier(path=_DEFAULT_CLASSIFIER_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze(image_path, classifier_path=_DEFAULT_CLASSIFIER_PATH):
    features = extract_features(image_path)

    if not os.path.exists(classifier_path):
        return {"features": features, "classifier_trained": False, "score": None}

    clf = load_classifier(classifier_path)
    score = float(clf.predict_proba(features.reshape(1, -1))[0, 1])
    return {"features": features, "classifier_trained": True, "score": score}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLIP-признаки + (если обучен) классификатор ИИ-генерации")
    parser.add_argument("image", help="Путь к изображению")
    parser.add_argument("--classifier", default=_DEFAULT_CLASSIFIER_PATH)
    args = parser.parse_args()

    print("Загружаю CLIP ViT-L/14 (при первом запуске скачивает веса, это долго)...")
    result = analyze(args.image, classifier_path=args.classifier)

    print(f"Вектор признаков: {result['features'].shape}, норма={np.linalg.norm(result['features']):.3f}")
    if result["classifier_trained"]:
        print(f"score={result['score']:.3f}")
    else:
        print("Классификатор ещё не обучен (нет файла весов) — датасет для обучения "
              "собирается на этапе экспериментов. Пока доступны только признаки CLIP.")

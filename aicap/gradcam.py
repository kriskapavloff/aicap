"""Grad-CAM для CLIP: показывает, какие зоны изображения повлияли на решение
классификатора — не только число, но и объяснение "почему".

ВАЖНО (честность, см. README): этот модуль зависит от двух вещей, которых
пока нет в проекте — скачанных весов CLIP (скачивание зависло на Xet-
протоколе HuggingFace, см. коммит cnn_detector.py) и обученного
классификатора (нужен датасет, см. experiments/). Код написан по спеку и
логически выверен, но вживую НЕ проверялся ни разу. Первая реальная
проверка — на этапе экспериментов, когда появится и то, и другое.

Технический нюанс (тоже отмечен в спеке заранее): классический Grad-CAM
(Selvaraju et al.) придуман для свёрточных сетей — там есть готовая
пространственная карта каналов в последнем conv-слое. CLIP ViT-L/14 —
Vision Transformer, пространственной карты каналов в этом смысле нет,
вместо неё — последовательность токенов, каждый из которых соответствует
патчу изображения (для ViT-L/14 при 224×224: патч 14×14, значит сетка
16×16=256 патчей + 1 CLS-токен). Реализация ниже — адаптация Grad-CAM на
эти патч-токены: те же формулы (градиент → усреднение по признакам →
взвешенная сумма → ReLU), только вместо "канал" везде "токен".
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from . import cnn_detector, preprocessing


def _find_transformer_blocks(visual_model):
    """Список transformer-блоков visual-энкодера CLIP.

    open_clip грузит ViT-L/14 либо своей нативной реализацией (атрибут
    .transformer.resblocks), либо через timm-бэкенд (атрибут .trunk.blocks) —
    судя по имени скачиваемого чекпоинта (timm/vit_large_patch14_clip_224),
    у нас, скорее всего, второй случай, но заранее не проверено, поэтому
    поддержаны оба варианта.
    """
    if hasattr(visual_model, "trunk") and hasattr(visual_model.trunk, "blocks"):
        return visual_model.trunk.blocks
    if hasattr(visual_model, "transformer") and hasattr(visual_model.transformer, "resblocks"):
        return visual_model.transformer.resblocks
    raise AttributeError(
        "Не удалось найти transformer-блоки в visual-энкодере CLIP — "
        "структура модели отличается от ожидаемой, нужно смотреть руками."
    )


def _register_hooks(block):
    state = {}

    def forward_hook(module, inputs, output):
        state["activations"] = output

    def backward_hook(module, grad_input, grad_output):
        state["gradients"] = grad_output[0]

    fwd_handle = block.register_forward_hook(forward_hook)
    bwd_handle = block.register_full_backward_hook(backward_hook)
    return state, fwd_handle, bwd_handle


def generate(image_path, classifier_path=None):
    """Возвращает 2D numpy-массив (сетка патчей, значения 0..1) — карту
    вклада каждой области изображения в решение классификатора."""
    classifier_path = classifier_path or cnn_detector._DEFAULT_CLASSIFIER_PATH

    model, clip_preprocess = cnn_detector._get_model()
    clf = cnn_detector.load_classifier(classifier_path)

    image = preprocessing.load_image(image_path)
    tensor = clip_preprocess(image).unsqueeze(0).to(cnn_detector._device)
    tensor.requires_grad_(True)

    blocks = _find_transformer_blocks(model.visual)
    state, fwd_handle, bwd_handle = _register_hooks(blocks[-1])

    try:
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)

        # Классификатор обучен через sklearn на numpy-признаках — переносим его
        # веса в тензор, чтобы посчитать логит как дифференцируемую операцию
        # и получить градиент через тот же граф, что и у CLIP.
        weight = torch.tensor(clf.coef_[0], dtype=features.dtype, device=features.device)
        bias = torch.tensor(clf.intercept_[0], dtype=features.dtype, device=features.device)
        logit = (features.squeeze(0) * weight).sum() + bias

        model.zero_grad()
        logit.backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    if "activations" not in state or "gradients" not in state:
        raise RuntimeError("Хуки не сработали — не удалось получить активации/градиенты последнего блока.")

    # (1, N+1, D): первый токен — CLS, остальные N — патчи. CLS отбрасываем,
    # он не соответствует конкретной области изображения.
    activations = state["activations"][0, 1:, :]
    gradients = state["gradients"][0, 1:, :]

    token_weights = gradients.mean(dim=1)  # аналог global-average-pool из оригинального Grad-CAM, но по токену
    cam = torch.relu((token_weights.unsqueeze(1) * activations).sum(dim=1))

    grid_size = int(round(cam.shape[0] ** 0.5))
    if grid_size * grid_size != cam.shape[0]:
        raise RuntimeError(f"Число патч-токенов ({cam.shape[0]}) не образует квадратную сетку — "
                            f"неожиданный размер входа или архитектура.")

    cam = cam.reshape(grid_size, grid_size)
    cam = cam / (cam.max() + 1e-8)

    return cam.detach().cpu().numpy()


def overlay_heatmap(image, cam, save_path, alpha=0.45):
    """image: PIL.Image (исходное, RGB). cam: 2D-массив 0..1 (сетка патчей),
    масштабируется до размера изображения и накладывается полупрозрачно."""
    heatmap = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)
    heatmap = np.asarray(heatmap, dtype=np.float32) / 255.0

    colored = plt.get_cmap("inferno")(heatmap)[:, :, :3]
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    blended = np.clip((1 - alpha) * base + alpha * colored, 0, 1)

    plt.figure(figsize=(5, 5))
    plt.imshow(blended)
    plt.axis("off")
    plt.title("Grad-CAM: зоны, повлиявшие на решение")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def explain(image_path, save_path, classifier_path=None):
    cam = generate(image_path, classifier_path)
    image = preprocessing.load_image(image_path)
    overlay_heatmap(image, cam, save_path)
    return cam


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Grad-CAM для CLIP-классификатора")
    parser.add_argument("image", help="Путь к изображению")
    parser.add_argument("--classifier", default=None)
    parser.add_argument("--plot-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    label = os.path.basename(args.image)
    save_path = os.path.join(args.plot_dir, f"{label}_gradcam.png")

    explain(args.image, save_path, classifier_path=args.classifier)
    print(f"Сохранено: {save_path}")

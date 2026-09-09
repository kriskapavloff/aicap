"""FFT-детектор: ищет отклонения от степенного закона распределения энергии
по частотам, характерного для естественных изображений.

Гипотеза (проверяется в experiments/): апсемплинг в генеративных моделях
вносит периодические артефакты в высокочастотной области спектра, из-за
которых радиальный профиль энергии хуже ложится на прямую в лог-лог осях.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_grayscale(image_path):
    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.float32)


def power_spectrum(gray):
    spectrum = np.fft.fft2(gray)
    spectrum = np.fft.fftshift(spectrum)
    return np.abs(spectrum) ** 2


def radial_profile(spectrum):
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)

    r_max = r.max()
    sums = np.bincount(r.ravel(), weights=spectrum.ravel(), minlength=r_max + 1)
    counts = np.bincount(r.ravel(), minlength=r_max + 1)
    return sums / np.maximum(counts, 1)


def fit_power_law(profile):
    # r=0 — постоянная составляющая (средняя яркость), в степенной закон не входит
    radii = np.arange(1, len(profile))
    values = profile[1:]

    valid = values > 0
    log_r = np.log(radii[valid])
    log_v = np.log(values[valid])

    slope, intercept = np.polyfit(log_r, log_v, deg=1)
    predicted = slope * log_r + intercept
    rmse = float(np.sqrt(np.mean((log_v - predicted) ** 2)))

    return slope, intercept, rmse


def anomaly_score(rmse, rmse_cap=3.0):
    return float(np.clip(rmse / rmse_cap, 0.0, 1.0))


def analyze(image_path, rmse_cap=3.0):
    gray = load_grayscale(image_path)
    spectrum = power_spectrum(gray)
    profile = radial_profile(spectrum)
    slope, intercept, rmse = fit_power_law(profile)

    return {
        "slope": slope,
        "intercept": intercept,
        "rmse": rmse,
        "score": anomaly_score(rmse, rmse_cap),
        "spectrum": spectrum,
        "profile": profile,
    }


def plot_spectrum(spectrum, save_path):
    plt.figure(figsize=(5, 5))
    plt.imshow(np.log1p(spectrum), cmap="inferno")
    plt.axis("off")
    plt.title("Спектр мощности (лог. шкала)")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_radial_profile(profile, slope, intercept, save_path):
    radii = np.arange(1, len(profile))
    values = profile[1:]
    valid = values > 0

    fitted = np.exp(intercept) * radii[valid].astype(np.float64) ** slope

    plt.figure(figsize=(6, 4))
    plt.loglog(radii[valid], values[valid], linewidth=1, label="наблюдаемый профиль")
    plt.loglog(radii[valid], fitted, "--", linewidth=1, label="степенной закон (подгонка)")
    plt.xlabel("радиус (частота)")
    plt.ylabel("энергия")
    plt.legend()
    plt.title("Радиальный профиль спектра мощности")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def _synthetic_test_image(size=256, seed=0):
    """Розовый шум (спектр ~1/f) — приближение статистики натуральных изображений,
    чтобы проверить пайплайн без реальной картинки под рукой."""
    rng = np.random.default_rng(seed)
    white_noise = rng.normal(size=(size, size))

    freqs = np.fft.fftfreq(size)
    fy, fx = np.meshgrid(freqs, freqs)
    radius = np.sqrt(fx**2 + fy**2)
    radius[0, 0] = radius[0, 0] if radius[0, 0] != 0 else 1e-6

    pink_filter = 1.0 / np.maximum(radius, 1e-6)
    spectrum = np.fft.fft2(white_noise) * pink_filter
    pink_noise = np.real(np.fft.ifft2(spectrum))

    normalized = (pink_noise - pink_noise.min()) / (pink_noise.max() - pink_noise.min())
    return (normalized * 255).astype(np.uint8)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="FFT-анализ изображения на признаки ИИ-генерации")
    parser.add_argument("image", nargs="?", help="Путь к изображению. Без аргумента — тест на синтетических данных")
    parser.add_argument("--rmse-cap", type=float, default=3.0)
    parser.add_argument("--plot-dir", default="results", help="Куда сохранить графики")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    if args.image:
        result = analyze(args.image, rmse_cap=args.rmse_cap)
        label = os.path.basename(args.image)
    else:
        print("Аргумент не передан — прогоняю на синтетическом изображении (розовый шум).")
        synthetic = _synthetic_test_image()
        gray = synthetic.astype(np.float32)
        spectrum = power_spectrum(gray)
        profile = radial_profile(spectrum)
        slope, intercept, rmse = fit_power_law(profile)
        result = {
            "slope": slope,
            "intercept": intercept,
            "rmse": rmse,
            "score": anomaly_score(rmse, args.rmse_cap),
            "spectrum": spectrum,
            "profile": profile,
        }
        label = "synthetic"

    print(f"slope={result['slope']:.3f}  intercept={result['intercept']:.3f}  "
          f"rmse={result['rmse']:.3f}  score={result['score']:.3f}")

    plot_spectrum(result["spectrum"], os.path.join(args.plot_dir, f"{label}_spectrum.png"))
    plot_radial_profile(
        result["profile"], result["slope"], result["intercept"],
        os.path.join(args.plot_dir, f"{label}_radial_profile.png"),
    )
    print(f"Графики сохранены в {args.plot_dir}/")

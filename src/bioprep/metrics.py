"""М4. Метрики качества первичной обработки.

На неделе 1 реализован PSNR; SSIM, F-мера и мера Пратта будут перенесены
из прототипа НИР вместе с эталонными тестами (V-модель, см. ADR 0001).
"""

import numpy as np


def psnr(reference: np.ndarray, result: np.ndarray, max_value: float = 1.0) -> float:
    """Пиковое отношение сигнал/шум, дБ: PSNR = 10 * lg(MAX^2 / MSE)."""
    reference = np.asarray(reference, dtype=np.float64)
    result = np.asarray(result, dtype=np.float64)
    if reference.shape != result.shape:
        raise ValueError("Изображения должны иметь одинаковую форму")
    mse = float(np.mean((reference - result) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(max_value**2 / mse)

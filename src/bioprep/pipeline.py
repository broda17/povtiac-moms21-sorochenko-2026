"""Конвейер первичной обработки: предобработка → М1 → М2 и М3 → постобработка.

Заготовка интерфейсов. Реализации модулей переносятся из прототипа НИР
по задачам бэклога (см. доску проекта).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PipelineResult:
    denoised: np.ndarray
    edges: np.ndarray
    segments: np.ndarray


def denoise(image: np.ndarray) -> np.ndarray:
    """М1: шумоподавление PCNN с параметрами, найденными PSO."""
    raise NotImplementedError("М1 будет перенесён из прототипа НИР")


def detect_edges(image: np.ndarray) -> np.ndarray:
    """М2: бинарная карта границ по карте феромона ACO."""
    raise NotImplementedError("М2 будет перенесён из прототипа НИР")


def segment(image: np.ndarray, edges: np.ndarray, levels: int = 3) -> np.ndarray:
    """М3: многоуровневая сегментация ГА по энтропии Капура + совмещение с картой границ."""
    raise NotImplementedError("М3 будет перенесён из прототипа НИР")


def run(image: np.ndarray) -> PipelineResult:
    """Полный конвейер; целевой бюджет — 500 мс на изображение 512×512."""
    clean = denoise(image)
    edges = detect_edges(clean)
    return PipelineResult(denoised=clean, edges=edges, segments=segment(clean, edges))

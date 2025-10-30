from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class Artifact:
    local_path: Path
    relative_key: str


def enumerate_artifacts(
    root: Path, includes: Iterable[str] | None = None
) -> List[Artifact]:
    if not root.exists() or not root.is_dir():
        return []

    patterns = list(includes or ["**/*"])
    items: list[Artifact] = []

    for pattern in patterns:
        for p in root.glob(pattern):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                items.append(Artifact(p, rel))

    items.sort(key=lambda a: a.relative_key)
    return items

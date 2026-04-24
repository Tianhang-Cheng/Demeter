from pathlib import Path

from setuptools import setup, find_namespace_packages

ROOT = Path(__file__).parent.resolve()
README = (ROOT / "readme.md").read_text(encoding="utf-8")

setup(
    name="demeter",
    version="0.1.0",
    description=(
        "Parametric model of crop plant morphology from 3D scans (ICCV 2025). "
        "Models plants as a graph of stem and leaf."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    author="Demeter project authors",
    url="https://tianhang-cheng.github.io/Demeter/",
    python_requires=">=3.10",
    packages=find_namespace_packages(
        where=".",
        include=["utils*", "representation*"],
    ),
    # PyTorch: install matching CUDA build separately, e.g.
    #   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: Other/Proprietary License",
    ],
)

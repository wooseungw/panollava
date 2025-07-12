"""
PanoLLaVA 설치 스크립트
"""

from setuptools import setup, find_packages
import os

# README 파일 읽기
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# requirements.txt 파일 읽기
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="panollava",
    version="1.0.0",
    author="PanoLLaVA Team",
    author_email="your.email@example.com",
    description="허깅페이스의 Image Encoder와 LLM 모델을 조합한 파노라마 이미지 멀티모달 AI 프레임워크",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/panollava",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
        ],
        "web": [
            "gradio>=3.0.0",
            "streamlit>=1.20.0",
            "flask>=2.0.0",
            "fastapi>=0.95.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.7.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "panollava-train=scripts.train:main",
            "panollava-eval=scripts.evaluate:main",
            "panollava-infer=scripts.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "panollava": [
            "config/*.yaml",
            "examples/*.py",
            "notebooks/*.ipynb",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/panollava/issues",
        "Source": "https://github.com/your-username/panollava",
        "Documentation": "https://github.com/your-username/panollava/wiki",
    },
    keywords=[
        "panorama",
        "multimodal",
        "llava",
        "vision-language",
        "transformers",
        "pytorch",
        "huggingface",
        "ai",
        "machine-learning",
        "computer-vision",
        "nlp",
    ],
)

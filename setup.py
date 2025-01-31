"""Setup configuration for LlamaHome package."""

from setuptools import find_packages, setup

# Core requirements that are needed for setup.py to run
setup_requires = [
    "setuptools>=45",
    "wheel",
]

# Main package requirements
install_requires = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "rich>=10.0.0",
    "tensorboard>=2.13.0",
    "plotly>=5.13.0",
    "pydantic>=2.0.0",
    "httpx>=0.25.0",
]

setup(
    name="llamahome",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    setup_requires=setup_requires,
    install_requires=install_requires,
    extras_require={
        "dev": [
            "black",
            "isort",
            "mypy",
            "ruff",
            "bandit",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
        ],
        "cuda": [
            "flash-attn>=2.3.0; platform_system!='Darwin' and python_version>='3.8'",
        ],
    },
    python_requires=">=3.8",
)

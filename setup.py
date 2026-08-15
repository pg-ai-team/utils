from setuptools import setup, find_packages

setup(
    name="pg-ai-utils",
    version="0.1.0",
    description="Shared training utilities for pg-ai team (wandb, config, env detection)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "wandb>=0.17",
        "torch>=2.0",
    ],
    extras_require={
        "local": ["python-dotenv"],
        "colab": ["kagglehub"],
    },
)

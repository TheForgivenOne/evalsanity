from setuptools import setup, find_packages

setup(
    name="evalsanity",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "evalsanity=evalsanity.cli.main:main",
        ],
    },
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "datasets>=2.0",
        "bitsandbytes>=0.43",
        "accelerate>=0.30",
    ],
    author="EvalSanity",
    description="Contamination-Resistant ML Evaluation Framework",
)

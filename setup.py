from setuptools import setup, find_packages

setup(
    name="AdaImgEncoder",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "wandb",
        "Pillow",
        "requests"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A variable-length image encoding system with similarity estimation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AdaImgEncoder",
)
from setuptools import setup, find_packages

setup(
    name="oncoterrain",
    version="0.1.0",
    description="CLI for OncoTerrain analysis",
    author="Vignesh Venkat",
    author_email="vvv11@scarletmail.rutgers.edu",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "click",
        "scanpy",
        "umap-learn",
        "gseapy",
        "matplotlib",
        "seaborn",
        "joblib",
        "numpy",
        "scikit-learn",
    ],
    include_package_data=True,
    package_data={
        "oncocli": ["OncoTerrain.joblib"],
    },
    entry_points={
        "console_scripts": [
            "oncoterrain = oncocli.oncocli:cli",
        ],
    },
)

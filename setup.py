from setuptools import setup, find_packages

setup(
    name="telecom_churn_project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
    "pandas==2.3.1",
    "numpy==2.3.1",
    "scikit-learn==1.7.1",
    "matplotlib==3.10.3",
    "seaborn==0.13.2",
    "jupyterlab==4.4.7",
    "notebook==7.4.5",
    "optuna==4.5.0",
    "category-encoders==2.8.1",
    "shap==0.48.0",
    "streamlit==1.50.0",
    "flask==3.1.2"
    ],
    python_requires=">=3.8",
    author="Hassan Lone",
    description="Telecom churn analysis project",
    url="https://github.com/Lone-Hassan/Telecom-Churn-Project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
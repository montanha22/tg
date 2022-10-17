from setuptools import setup

setup(
    name="tg",
    version="0.0.1",
    entry_points={
        "console_scripts": [
            "init_tg = tg.__init__:change_mlflow_yaml",
        ]
    },
     include_package_data=True
)

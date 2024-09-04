from distutils.core import setup

ext_modules = []
cmdclass = {}

setup(
    name="tabular_prediction",
    version="1.0.0",
    description="",
    url="https://github.com/yulun-rayn/TabularPrediction",
    author="Yulun Wu",
    author_email="yulun_wu@berkeley.edu",
    license="MIT",
    packages=["tabular_prediction"],
    cmdclass=cmdclass,
    ext_modules=ext_modules
)

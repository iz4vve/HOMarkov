from setuptools import setup, find_packages

setup(
    name="HOMarkov",
    version="0.4.2",
    url="https://github.com/iz4vve/HOMarkov",
    download_url="",
    author="Pietro Mascolo",
    author_email="iz4vve@gmail.com",
    description="High Order Markov Chain model",
    packages=find_packages(exclude=["tests"]),
    zip_safe=False,
    include_package_data=True,
    platforms="any",
    license="GPL3",
    install_requires=[
        "numpy>=1.13.0",
        "scikit-learn>=0.18.1",
        "pandas>=0.20.0",
    ]
)

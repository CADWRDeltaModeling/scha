from setuptools import setup, find_packages
import versioneer

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(
    name="scha",
    version=versioneer.get_version(),
    license="MIT",
    cmdclass=versioneer.get_cmdclass(),
    description="State-space cyclical harmonic analysis (SCHA) tools",
    long_description=readme,
    long_description_content_type="text/x-rst",
    install_requires=[],
    include_package_data=True,
    packages=find_packages("src"),
    package_dir={"": "src"},
    author="Eli Ateljevich",
    author_email="Eli.Ateljevich@water.ca.gov",
    url="",
    keywords=["scha", "state-space", "time-series"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)

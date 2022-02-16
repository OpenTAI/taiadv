import pathlib

from setuptools import find_packages, setup

# The directory containing this file
# HERE = pathlib.Path(__file__)

# The text of the README file
README = pathlib.Path('README.md').read_text()

# This call to setup() does all the work
setup(
    name='taiadv',
    version='0.0.1',
    description='Adversarial Attack and Defense Toolbox and Benchmark',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/OpenTAI/taiadv',
    author='OpenTAI',
    author_email='',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=['torch', 'torchvision'],
)

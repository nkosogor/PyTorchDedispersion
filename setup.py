from setuptools import setup, find_packages

setup(
    name='PyTorchDedispersion',
    version='0.1.0', 
    author='Nikita Kosogorov',
    author_email='nakosogorov@gmail.com',
    description='A package for detecting radio signal candidates using dedispersion with PyTorch.',
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'numpy',
        'matplotlib',
        'your'  
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    license='BSD 3-Clause License'
)

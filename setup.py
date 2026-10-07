from setuptools import setup
import io

setup(
    name='spliceai2',
    packages=['spliceai2'],
    version='2.0',
    description='A unified framework for quantitative splicing and transcript prediction',
    long_description=io.open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Illumina/SpliceAI2',
    license='SpliceAI2 Model Terms of Use',
    license_files=['LICENSE'],
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.21',
        'pandas>=1.5',
        'pyfaidx>=0.7',
        'torch>=2.1',
        'lightning>=2.1'
    ],
    entry_points={'console_scripts': ['spliceai2=spliceai2.score:main']},
    author='Kishore Jaganathan',
    author_email='kjaganathan@illumina.com'
)

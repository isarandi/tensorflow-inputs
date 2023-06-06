from setuptools import setup

setup(
    name='tensorflow-inputs',
    version='0.1.0',
    author='István Sárándi',
    author_email='istvan.sarandi@uni-tuebingen.de',
    packages=['tensorflow_inputs'],
    license='LICENSE',
    description='Input image data sources wrapped as TensorFlow Datasets',
    python_requires='>=3.8',
    install_requires=['tensorflow', 'opencv-python', 'imageio', 'more-itertools', 'Pillow'],
)

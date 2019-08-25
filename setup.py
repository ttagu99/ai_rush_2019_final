#nsml: nsml/ml:cuda9.0-cudnn7-tf-1.11torch1.0keras2.2

from distutils.core import setup
setup(
    name='airush1',
    version='1.0',
    install_requires=[
            'tqdm',
            'torch>=1.0',
            'lightgbm',
            #'TensorFlow >= 1.12.0',
            'scikit-image',
            'keras_applications >= 1.0.7',
            'efficientnet==0.0.4'
            #'pickle-mixin',
            #'torchvision',
            #'pandas>=0.24.0',
            #'scikit-multilearn',
            #'cnn_finetune',
            #'efficientnet-pytorch',
            #'torchsummary',
            #'Pillow'
            #'iterative-stratification'
    ]
)

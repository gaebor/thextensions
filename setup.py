from setuptools import setup

setup(name='thextensions',
    version='0.1',
    description='Some theano extensions.',
    url='https://github.com/gaebor/thextensions',
    author='Gabor Borbely',
    author_email='borbely@math.bme.hu',
    license='MIT',
    packages=['thextensions'],
    install_requires=['theano', 'numpy'],
    keywords='theano machine learning optimization gradient descent hessian',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules ',
    ],
    zip_safe=False)

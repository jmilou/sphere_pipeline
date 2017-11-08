from setuptools import setup, find_packages

setup(
    name='sphere_pipeline',
    version='0.1',
    description='Pipeline to analyze SPHERE images from irdis or zimpol',
    url='https://github.com/jmilou/sphere_pipeline',
    author='Julien Milli',
    author_email='jmilli@eso.org',
    license='MIT',
    keywords='image processing data analysis',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'astropy', 'pandas', 'matplotlib','pandas','datetime'
    ],
    zip_safe=False
)

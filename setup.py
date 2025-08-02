from setuptools import setup, find_packages

setup(
    name='sdf_geometry',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pyvista',
    ],
    author='Soliman Villapando',
    description='Composable SDFs for geometric modeling and optimization in CSDL',
    url='https://github.com/svillapando/sdf_geometry',
)

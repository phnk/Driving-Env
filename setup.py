'''
Performs packaging for pip installation. 
'''
from setuptools import setup

setup(
    name="gym_driving", 
    version='0.0.1', 
    install_requires=['gym', 'pybullet', 'numpy', 'matplotlib'],
    py_modules=[]
    )

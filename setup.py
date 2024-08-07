from setuptools import setup, find_packages

setup(
	name='blang',
	version='0.0.1',
	description='teddysum machine learning tools based on bllossom',
	url='https://github.com/teddysum/bLang',
	author='teddysum',
	author_email='ybjeong@teddysum.ai',
	packages=find_packages(),
	install_requires=['pandas']
)
from setuptools import setup, find_packages

setup(
    name='enki-client',
    version='0.1',
    description='A client for inference services provided by enki.farm',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Marco Crisafulli',
    author_email='marco.crisafulli@enki.farm',
    url='https://github.com/enki-farm/enki-client',
    packages=find_packages(),
    install_requires=[
        'kserve',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
    ],
    python_requires='>=3.9, <3.12',
)
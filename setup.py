from setuptools import setup
import os

currentPath = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(currentPath, 'README.rst'), 'r') as f:
    long_description = f.read()

long_description = '\n' + long_description
setup(name='pypdn',
      version='1.0.0',
      description='Python package to read and write Paint.NET (PDN) images.',
      long_description=long_description,
      author='Addison Elliott',
      author_email='addison.elliott@gmail.com',
      url='https://github.com/addisonElliott/pypdn',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: MIT License',
          "Programming Language :: Python",
          'Programming Language :: Python :: 3',
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Operating System :: OS Independent"

      ],
      keywords='paint dot net paintdotnet dotnet .net pdn library read write save layered images image editor',
      project_urls={
          'Documentation': 'https://github.com/addisonElliott/pypdn',
          'Source': 'https://github.com/addisonElliott/pypdn',
          'Tracker': 'https://github.com/addisonElliott/pypdn/issues',
      },
      python_requires='>=3',
      py_modules=['polarTransform'],
      license='MIT License',
      install_requires=[
          'numpy', 'scikit-image', 'aenum']
      )

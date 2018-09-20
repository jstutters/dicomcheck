"""Setup file for benefitregsub."""

from setuptools import setup

setup(name='dicomcheck',
      version='0.1.0',
      description='MRI sequence comparison tool',
      author='Jon Stutters',
      author_email='j.stutters@ucl.ac.uk',
      url='http://wiki.mstrials.ion.ucl.ac.uk',
      packages=[
          'dicomcheck'
      ],
      include_package_data=True,
      zip_safe=True,
      install_requires=[
          'numpy',
          'pydicom>=1.0.0'
      ],
      entry_points={
          'console_scripts': [
              'dicomcheck = dicomcheck.main:main'
          ]
      })

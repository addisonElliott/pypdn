Introduction
=================
pypdn is a Python package for reading and writing Paint.NET (PDN) images. "Paint.NET is image and photo editing software for PCs that run Windows. It features an intuitive and innovative user interface with support for layers, unlimited undo, special effects, and a wide variety of useful and powerful tools. An active and growing online community provides friendly help, tutorials, and plugins."

When using Paint.NET, the default file format that the images are saved in are PDN's, which is a proprietary format Paint.NET uses. The benefit of this format over BMP, PNG, JPEG, etc is that it stores layer information and properties that are not present in traditional image formats.

You can get Paint.NET `here <https://www.getpaint.net/>`_.

Installing
=================
Prerequisites
-------------
* Python 3
* Dependencies:
    * numpy
    * scikit-image
    * aenum

Installing pypdn
-------------------------
pypdn is currently available on `PyPi <https://pypi.python.org/pypi/pypdn/>`_. The simplest way to
install alone is using ``pip`` at a command line::

  pip install pypdn

which installs the latest release.  To install the latest code from the repository (usually stable, but may have
undocumented changes or bugs)::

  pip install git+https://github.com/addisonElliott/pypdn.git


For developers, you can clone the pypdn repository and run the ``setup.py`` file. Use the following commands to get
a copy from GitHub and install all dependencies::

  git clone pip install git+https://github.com/addisonElliott/pypdn.git
  cd pypdn
  pip install .

or, for the last line, instead use::

  pip install -e .

to install in 'develop' or 'editable' mode, where changes can be made to the local working code and Python will use
the updated polarTransform code.

Test and coverage
=================
To test the code on any platform, make sure to clone the GitHub repository to get the tests and run the following from
the repository directory::

  python -m unittest discover tests

Example
=================
TODO Fix this up
Input image:

.. image:: http://polartransform.readthedocs.io/en/latest/_images/verticalLines.png
    :alt: Cartesian image

.. code-block:: python

    import polarTransform
    import matplotlib.pyplot as plt
    import imageio

    verticalLinesImage = imageio.imread('IMAGE_PATH_HERE')

    polarImage, ptSettings = polarTransform.convertToPolarImage(verticalLinesImage, initialRadius=30,
                                                                finalRadius=100, initialAngle=2 / 4 * np.pi,
                                                                finalAngle=5 / 4 * np.pi)

    cartesianImage = ptSettings.convertToCartesianImage(polarImage)

    plt.figure()
    plt.imshow(polarImage, origin='lower')

    plt.figure()
    plt.imshow(cartesianImage, origin='lower')

Resulting polar domain image:

.. image:: http://polartransform.readthedocs.io/en/latest/_images/verticalLinesPolarImage_scaled3.png
    :alt: Polar image

Converting back to the cartesian image results in:

.. image:: http://polartransform.readthedocs.io/en/latest/_images/verticalLinesCartesianImage_scaled.png
    :alt: Cartesian image

Roadmap & Bugs
=================
- Write docstrings and create basic documentation for NRBF and PDN classes
- Fix issue where _id and _classID are mixed up in NRBF reader
- Add support for writing NRBF files
- Write unit tests for NRBF
- Add support for writing PDN files (must complete writing NRBF files first)

Pull requests are welcome (and encouraged) for any or all issues!

License
=================
pypdn has an MIT-based `license <https://github.com/addisonElliott/pypdn/blob/master/LICENSE>`_.

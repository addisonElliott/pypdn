Introduction
=================
pypdn is a Python package for reading and writing Paint.NET (PDN) images.

"Paint.NET is image and photo editing software for PCs that run Windows. It features an intuitive and innovative user interface with support for layers, unlimited undo, special effects, and a wide variety of useful and powerful tools. An active and growing online community provides friendly help, tutorials, and plugins."

When using Paint.NET, the default file format that the images are saved in are PDN's, which is a proprietary format Paint.NET uses. The benefit of this format over BMP, PNG, JPEG, etc is that it stores layer information and properties that are not present in traditional image formats.

You can get Paint.NET `here <https://www.getpaint.net/>`_.

Paint.NET is developed using C# (hence the .NET). Besides a basic XML header at the beginning, it primarily uses the
BinaryFormatter class in C# to save the relevant classes when saving an image. This uses the `NRBF protocol
<https://msdn.microsoft.com/en-us/library/cc236844.aspx>`_. A custom reader was developed to read the NRBF file and
then this library essentially just parses the data from NRBF into more readable and user-friendly format. You can
access the NRBF reader from the pypdn module as well in case you have any use for it.

Installing
=================
Prerequisites
-------------
* Python 3.4+
* Dependencies:
    * numpy
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
the updated code.

Test and coverage
=================
To test the code on any platform, make sure to clone the GitHub repository to get the tests and run the following from
the repository directory::

  python -m unittest discover -v tests

Example
=================
For the below example, any PDN file will do. If you are looking for an example, check out the tests/data directory for
some!

.. code-block:: python

    import pypdn
    import matplotlib.pyplot as plt

    layeredImage = pypdn.read('Untitled3.pdn')
    print(layeredImage)
    # Contains width, height, version and layers of the image within the class
    # Version being the Paint.NET version that the image was saved with

    # Each layer contains the name, visibility boolean, opacity (0-255), isBackground and blendMode
    # From what I can tell, the isBackground property is not that useful
    # The blend mode is how the layer should be blended with the layers below it
    # These attributes are loaded from the PDN file but can be edited in the code as well
    print(layeredImage.layers)
    layer = layeredImage.layers[0]
    layer.visible = True
    layer.opacity = 255
    layer.blendMode = pypdn.BlendType.Normal

    layer = layeredImage.layers[1]
    layer.visible = True
    layer.opacity = 161
    layer.blendMode = pypdn.BlendType.Additive

    # Finally, the most useful thing is being able to combine the layers and flattn them into one image
    # Call the flatten function to do so
    # It will go through each layer and apply them IF the visibility is true!
    # The layer opacity and blend mode will be taken into effect
    #
    # The flattened image is a RGBA Numpy array image
    # The asByte parameter determines the data type of the flattened image
    # If asByte is True, then the dtype will be uint8, otherwise it will be a float in range (0.0, 1.0)
    flatImage = layeredImage.flatten(asByte=True)

    plt.figure()
    plt.imshow(flatImage)

    # Individual layer images can be retrieved as well
    # Note: This does NOT apply blending or the layer opacity
    # Rather, it is the image data that is saved by Paint.NET for the layer
    plt.figure()
    plt.imshow(layeredImage.layers[1].image)

    plt.show()

Using the Untitled3.pdn in the tests/data directory, this is the text output:

.. code-block::

    >>> print(layeredImage)
    pypdn.LayeredImage(width=800, height=600, version=System_Version(Major=4, Minor=21, Build=6589, Revision=7045), layers=[pypdn.Layer(name=Background, visible=True, isBackground=True, opacity=255, blendMode=<BlendType.Normal: 0>), pypdn.Layer(name=Layer 2, visible=True, isBackground=False, opacity=161, blendMode=<BlendType.Additive: 2>)])

    >>> print(layeredImage.layers)
    [pypdn.Layer(name=Background, visible=True, isBackground=True, opacity=255, blendMode=<BlendType.Normal: 0>), pypdn.Layer(name=Layer 2, visible=True, isBackground=False, opacity=161, blendMode=<BlendType.Additive: 2>)]

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

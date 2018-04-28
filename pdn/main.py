# This file should not be tracked by Git.
# It is just meant for basic testing and such to make sure API is working

from pdn import nrbf
from pdn import nrbf2
import json
from pdn.namedlist import namedlist

# I want to first begin by testing the nrbf class and cleaning it up so I understand it
# Also, I would like to extend it to WRITE as well, which is not currently the case

# Also, it would be good to test this class with the reading and writing components of it
# Probably nothing too extensive, simple read and check values
# Save those to a new file and check that they are exactly the same.

# filename = 'D:/Users/addis/Desktop/Untitled.pdn'
filename = '../tests/data/imageRawNRBF'
# filename = '../tests/data/arraysSerialized'
# filename = 'D:/Users/addis/Desktop/Untitled4.pdn'
# filename = 'D:/Users/addis/Desktop/Untitled2.pdn'
# filename = 'D:/Users/addis/Desktop/Untitled3.pdn'

with open(filename, 'rb') as fh:
    x = nrbf2.NRBF(fh)

# with open(filename, 'rb') as fh:
#     serial = nrbf.Serialization(fh)
#     data = serial.read_stream()
# #
#     json_encoder = nrbf.JSONEncoder(indent=4, check_circular=False)
#     print(json_encoder.encode(data))
#     print(data)
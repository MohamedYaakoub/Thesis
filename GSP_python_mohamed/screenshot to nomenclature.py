from PIL import Image
import numpy
from numpy import asarray


image = Image.open('images/nomenc-1.jpg')

numpydata = asarray(image)

# <class 'numpy.ndarray'>
print(type(numpydata))

#  shape
print(numpydata.shape)

with open("random_array.txt", "ab") as f:
    for i in range(3):
        numpy.savetxt(f, numpydata[:,:,i])
        f.write(b"\n")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
# First, load the image
filename = "flower.jpg"
image = mpimg.imread(filename)

# Print out its shape
print(image.shape)
plt.imshow(image)
plt.show()

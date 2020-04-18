import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show_img(img, size):
    plt.figure(figsize=(size,size))
    # unnormalize
    img = img / 2 + 0.5  
    npimg = img.numpy()
    npimg = np.clip(npimg, 0., 1.)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show() 

def show_images(dataset, classes, size):
    # get some random training images
    dataiter = iter(dataset)
    images, labels = dataiter.next()

    # show images
    show_img(torchvision.utils.make_grid(images), size)


def show_random_images(dataset, classes, size):
	# get some random training images
	dataiter = iter(dataset)
	images, labels = dataiter.next()

	img_list = range(5, 10)

	# show images
	print('shape:', images.shape)
	show_img(torchvision.utils.make_grid(images[img_list]), size)
	# print labels
	print(' '.join('%5s' % classes[labels[j]] for j in img_list))

  
import numpy as np
import csv
import matplotlib.pyplot as plt
from kernelPCA_Class import Gaussian_Kernel
from sklearn.decomposition import PCA
import pdb
from random import shuffle

class kPCA_usps():
    def __init__(self):
        self.name='usps_example'
        self.data_labels, self.data=self.readData('USPS_dataset//zip.train')
        self.training_images = self.extractDataSets(self.data, self.data_labels, 300)
        self.test_labels, self.test_images=self.readData('USPS_dataset//zip.test')
        #self.test_images = self.extractDataSets(self.test_data, self.test_labels, 50)
        #self.test_images=self.shuffleListAndExtract(self.test_images,50)
        self.gaussian_images=self.addGaussianNoise(np.copy(self.test_images))
        self.speckle_images=self.addSpeckleNoise(np.copy(self.test_images))
        self.kPCA_gaussian=Gaussian_Kernel()
        self.C=0.5

    def readData(self,filePath):
        labels=[]
        images=[]
        with open(filePath, 'r') as f:
            content = f.readlines()
            for index, pixels in enumerate(content):
                # split string of floats into array
                pixels = pixels.split()
                # the first value is the label
                label = int(float(pixels[0]))
                # the reset contains pixels
                pixels = -1*np.array(pixels[1:],dtype=float)    #flips black => white so numbers are black, background white
                # Reshape the array into 16 x 16 array (2-dimensional array)
                labels.append(label)
                images.append(pixels)
        return np.asarray(labels),np.asarray(images)

    def extractDataSets(self, data, labels, nEach):
        shuffleIndices = np.arange(data.shape[0])
        shuffle(shuffleIndices)
        data = data[shuffleIndices, :]
        numbers = 10
        number_label = np.ones((10, 1)) * (nEach -1)
        retVal = np.zeros((nEach*numbers, data.shape[1]), dtype=float)
        count = 0
        for t,i in enumerate(labels):
            if (number_label[i] < 0):
                continue
            number_label[i] -= 1
            retVal[count, :] = data[t, :]
            count += 1
        return retVal

    def shuffleListAndExtract(self,list,noElements):
        noImages=list.shape[0]
        shuffleIndicies=np.arange(noImages)
        shuffle(shuffleIndicies)
        list=list[shuffleIndicies]
        list=list[:noElements]
        return list

    def addSpeckleNoise(self, images):
        noisy_images=[]
        for image in images:
            p=0.4
            pixels_speckle_noise_push = np.array([np.random.choice([-1., num, 1.], p=[p / 2., 1 - p, p / 2.]) for num in image])
            noisy_images.append(pixels_speckle_noise_push)
        return noisy_images

    def addGaussianNoise(self,images):
        noisy_images=[]
        for row in images:
            pixels=row
            '''add noise here '''
            mu = -1
            sigma = 0.5
            pixels_plus_noise = pixels + np.random.normal(loc=mu,scale=sigma,size=256)
            index=np.where(abs(pixels_plus_noise)>1)
            pixels_plus_noise[index]=np.sign(pixels_plus_noise[index])
            noisy_images.append(pixels_plus_noise)
        return np.asarray(noisy_images)

    def display(self,images):
        for image in images:
            image = image.reshape((16, 16))
            plt.imshow(image,'gray',interpolation='none')
            #plt.show()

    def kernelPCA_gaussian(self, max_eigVec_lst, threshold):
        # create Projection matrix for all test points and for each max_eigVec
        kGram, norm_vec = self.kPCA_gaussian.normalized_eigenVectors(self.training_images, self.C)
        projection_kernel = self.kPCA_gaussian.projection_kernel(self.training_images, self.test_images, self.C)
        projection_matrix_centered = self.kPCA_gaussian.projection_centering(kGram, projection_kernel)
        result_lst=[]
        for max_eigVec in max_eigVec_lst:
            reconstructed_images = []
            print(max_eigVec)
            projection_matrix = np.dot(projection_matrix_centered, norm_vec[:, :max_eigVec])

            # approximate input
            gamma = self.kPCA_gaussian.gamma_weights(norm_vec, projection_matrix, max_eigVec)
            # np.random.seed(20)
            # z_init = np.random.rand(self.nClusters * self.nTestPoints, self.nDim)
            z_init = np.copy(self.gaussian_images)  # according to first section under chapter 4,
            # in de-noising we can use the test points as starting guess
            z_init_old = np.zeros(z_init.shape)
            #for tp in range(len(self.test_images)):
            for tp in range(0,1):
                max_distance = 1
                while max_distance > threshold:
                    try:
                        approx_z = self.kPCA_gaussian.approximate_z_single(gamma[tp, :], z_init[tp, :], self.training_images,
                                                                           self.C, 256)
                    except ValueError:
                        print("zero denominator! Initializing with previous z_init.")
                        approx_z = self.kPCA_gaussian.approximate_z_single(gamma[tp, :], z_init[tp-1, :], self.training_images,
                                                                           self.C, 256)
                    max_distance = (np.linalg.norm(z_init[tp, :] - approx_z, axis=1, ord=2))
                    z_init[tp, :] = approx_z

                reconstructed_images.append(approx_z)
            result_lst.append(reconstructed_images)
        return result_lst


if __name__ == "__main__":
    usps=kPCA_usps()
    ax1 = plt.subplot(611)
    #usps.display(usps.test_images[0:1], ax1)
    usps.display(usps.gaussian_images[0:1])
    print(usps.test_labels[0])
    for i in range(5):
        ax2 = plt.subplot("21%d" % (i+1))
        reconstruted_images=usps.kernelPCA_gaussian([0],0.1)
        usps.display(reconstruted_images[0])
        plt.show()
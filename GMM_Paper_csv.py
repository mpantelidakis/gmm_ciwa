from numpy import mat
import pandas as pd
from GMM import *
import util as util
from scipy import stats
import glob
import argparse
# from matplotlib import pyplot as plt
from PIL import Image


# import sys
# np.set_printoptions(threshold=sys.maxsize)

class GMM_Paper:

    def GMM_ciwa(self, csv_path, No_Component=3):

        data = pd.read_csv(csv_path, header = 0)
        data = data.reset_index()

        col="Temp(c)"
        temperature=data[[col]]
        temperature=np.array(temperature)
        
        minimum=abs(min(temperature))
        temperature=temperature+minimum
        maximum=max(temperature)
        temperature=temperature/maximum+1e-4


        # Heatmap for debugging
        # plt.imshow(temperature, cmap='hot', interpolation='nearest')
        # plt.show()
        #-----------------------------

        x_rgb=np.array(data[["R","G","B"]])
        x_hsv=util.rgb_to_hsv(x_rgb)
        Tl_prime = []
        mu_guess = np.array([70,0.67,0.53]) #CARE! Change these values (dataset-specific HSV means). Arguments 2 and 3 are between 0-1.
        sigma_guess = np.array([[30,0,0],[0,0.5,0],[0,0,0.25]])**2 #CARE! do not change these values, as they are the ones in the GMM paper.

        #  https://stats.stackexchange.com/questions/331283/how-to-calculate-the-probability-of-a-data-point-belonging-to-a-multivariate-nor?fbclid=IwAR318YmXyipd5XfP4BMnYvlvd7Ce_wakTSIYJXntd0rHymJx1_pdQnQ3UXM
        for id_no, pixel in enumerate(x_hsv):
            #  Mahalanobis distance
            m_dist_x = np.dot((pixel-mu_guess).transpose(),np.linalg.inv(sigma_guess))
            m_dist_x = np.dot(m_dist_x, (pixel-mu_guess))

            # Add leaf pixels to the Tl_prime set
            if (1-stats.chi2.cdf(m_dist_x, 3)) > 0.1: # 90% clustering probability
                Tl_prime.append(id_no)

        Tl_prime_mean=np.mean(temperature[Tl_prime]) # Mean temp of leaf pixels

        #-----------------------------
        initial_mu=[Tl_prime_mean,(-30+minimum)/maximum+1e-4,(90+minimum)/maximum+1e-4] #CARE! do not change these values, as they are the ones in the GMM paper.
        initial_sigma=[100,100,100] #CARE! do not change these values, as they are the ones in the GMM paper.
        gmm = GaussianMixModel(temperature,No_Component, Tl_prime_mean, initial_mu, initial_sigma)
        gmm.fit(10)

        # Uncomment the following line to visualize the GMM.
        # util.plot_1D(gmm,temperature,col)

        Tl_prime_prime = []
        for id_no, clustprop in enumerate(gmm.Z):
            if (clustprop[0,0]>clustprop[0,1]) and (clustprop[0,0]>clustprop[0,2]):
                Tl_prime_prime.append(id_no)
                
       
        T = np.intersect1d(Tl_prime, Tl_prime_prime) #The IDs of the intersection pixels

        # Generate a mask of 60x80. The mask has a value of 1 for pixels in the T set and 0 for the rest.
        np_mask = np.zeros(4800,)
        np_mask[T] = 1
        np_mask = np_mask.reshape(60,80)

        
        newshape = np_mask.shape + (1,)
        np_mask = np_mask.reshape(newshape)

        np_image = np.zeros([60,80,3],dtype=np.uint8)
        np_image[:] = [165, 42, 42]

        np_image = np.where(np_mask, [0, 255, 0], [165, 42, 42])

        prediction = Image.fromarray(np_image.astype(np.uint8))
        prediction = prediction.resize((480, 320))
        prediction.save("Testset/test/" + csv_path.split('/')[2].replace(".csv",'.jpg'))
 
class SmartFormatter(argparse.HelpFormatter):


    def _split_lines(self, text, width):

        if text.startswith('R|'):

            return text[2:].splitlines()  

        return argparse.HelpFormatter._split_lines(self, text, width)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform Gaussian mixture models over images.', formatter_class=SmartFormatter)
    parser.add_argument('-act', '--actions', help='R|Perform all available actions for all images.',required=False,  action='store_true')
    parser.add_argument('-i', '--input', type=str, help='Input image. Ex. img.jpg', required=False)
    args = parser.parse_args()
    
    gmm = GMM_Paper()
    
    if args.actions:
        gmm.csv_path_list = glob.glob("data/csv_files/*.csv")
        for csv_path in gmm.csv_path_list:
            gmm.GMM_ciwa(csv_path.replace("\\","/"))
        print("Total number of images: ",len(gmm.csv_path_list))
    
    elif args.input:
        gmm.GMM_ciwa("data/csv_files/"+ args.input)
       
    


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import numpy as np


def plot_latent_MNIST(autoencoder, data, num_batches=100):
    '''
    Plots 2D latent space z as a scatter plot
    '''
    print("Z-space scatter plot")
    for i, (x, y) in enumerate(data):
        mu, sigma = autoencoder.encoder(x.to(device).view(-1,784))
        z = mu + sigma * torch.distributions.Normal(0,1).sample(mu.shape).to(device)
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show()

def plot_reconstructed_MNIST(autoencoder, data, n=15):
    '''
    Plots actual and reconstructed digits as image output of the autoencoder. Default number of digits shown is 15.
    '''
    w = 28
    img = np.zeros((2*w, n*w))
    print("Upper row represents the original image and below is its reconstruction")
    for i , (inputs, _) in enumerate(data):
        recon_imgs, _, _, _ = autoencoder(inputs.view(-1, inputs.shape[-1] * inputs.shape[-2]).to(device))
        recon_img = recon_imgs[0, :].reshape(28, 28).to('cpu').detach().numpy()
        img[:w, i*w:(i+1)*w] = inputs[0]
        img[w:2*w, i*w:(i+1)*w] = recon_img
        if i==n-1: break
    plt.imshow(img)
    plt.show()

def plot_reconstructed_ped_dist(autoencoder, data, ped_loc_mean, ped_loc_max):
    '''
    Plots Actual and reconstructed pedestrian distribution as a scatter plot, given the autoencoder.
    '''
    print("Actual Points")
    for i, (inputs,_) in enumerate(data):
        inputs = inputs.squeeze(2).cpu().detach().numpy()
        inputs = np.stack([inputs[:,0] * ped_loc_max[0], inputs[:,1] * ped_loc_max[1]], axis=-1)
        inputs += ped_loc_mean
        plt.scatter(inputs[:,0], inputs[:,1], color='blue')
    plt.axis('equal')
    plt.show()

    print("Reconstructed Points")
    for i, (inputs,_) in enumerate(data):
        recon_points, _, _, _ = autoencoder(inputs.view(-1, inputs.shape[-1] * inputs.shape[-2]).to(device))
        #print(recon_points.shape)
        recon_points = recon_points.cpu().detach().numpy()
        # Reverse-Normalizing from [-1,1] to actual coordinate space
        recon_points = np.stack([recon_points[:,0] * ped_loc_max[0], recon_points[:,1] * ped_loc_max[1]], axis=-1)
        recon_points += ped_loc_mean
        plt.scatter(recon_points[:,0], recon_points[:,1], color='red')
    plt.axis('equal')
    plt.show()
    


def plot_generated_ped_dist(autoencoder, ped_loc_mean, ped_loc_max ,n=1000, plot=True):
    '''
    Plots the generated pedestrian distribution upon sampling the latent vector z from the Standard Normal Distribution.
    '''
    z = torch.distributions.Normal(0,1).sample((n,autoencoder.latent_dims)).to(device)
    gen_points = autoencoder.decoder(z)
    gen_points = gen_points.cpu().detach().numpy()

    # Reverse Normalizing to fit on original coordinates
    gen_points = np.stack([gen_points[:,0] * ped_loc_max[0], gen_points[:,1] * ped_loc_max[1]], axis=-1) 
    gen_points += ped_loc_mean
    
    count = 0
    for point_num in range(len(gen_points)):
        if gen_points[point_num, 0]>130 and gen_points[point_num, 0]<150 and gen_points[point_num,1]>50 and gen_points[point_num,1]<70:
            count+=1
    if plot:
        plt.scatter(gen_points[:,0], gen_points[:,1], color='green')
        plt.gca().add_patch(Rectangle((130, 50), 20, 20, facecolor='none', ec='orange', lw=1))
        plt.axis('equal')
        plt.show()
        print(f"No of pedestrians  in critical area: {count}")
    return count, gen_points


def plot_generated_MNIST(autoencoder,data, n=15):
    '''
    Plots generated digits as images upon sampling latent vector z from the standard normal distribution.
    '''
    w = 28
    img = np.zeros((w, n*w))
    print("Generated Images")
    z = torch.distributions.Normal(0,1).sample((n, autoencoder.latent_dims)).to(device)
    gen_digits = autoencoder.decoder(z).cpu().detach().numpy().reshape(-1, 28, 28)
    for i in range(n):
        img[:, i*w:(i+1)*w] = gen_digits[i,...]
    plt.imshow(img)
    plt.show()
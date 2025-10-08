from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import numpy as np

class NodePCA(nn.Module):
    def __init__(self, n_components:int=1, path=None):
        super(NodePCA, self).__init__()

        """
        n_components: int or float
        """

        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

        if path is not None:
            self.load(path)
    
    def train(self, data):
        """
        data: [batch, n_params]
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        data_mean = data.mean(axis=0) # [n_params]
        data_centered = data - data_mean
        coeff = self.pca.fit_transform(data_centered) # [n_leaf, n_components]

        coeff_mean = coeff.mean(axis=0)  # [n_components]
        coeff_std = coeff.std(axis=0)  # [n_components]
        # coeff_lower = coeff_mean - 3 * coeff_std
        # coeff_upper = coeff_mean + 3 * coeff_std

        self.register_buffer('coeff_mean', torch.from_numpy(coeff_mean).float().cuda()) # [n_components]
        self.register_buffer('coeff_std', torch.from_numpy(coeff_std).float().cuda()) # [n_components]

        self.register_buffer('data_mean', torch.from_numpy(data_mean).float().cuda())
        self.register_buffer('mean', torch.from_numpy(self.pca.mean_).float().cuda()) # [n_params]
        self.register_buffer('components', torch.from_numpy(self.pca.components_).float().cuda()) # [n_components, n_params]
        self.register_buffer('explained_variance', torch.from_numpy(self.pca.explained_variance_).float().cuda())
        self.register_buffer('explained_variance_ratio', torch.from_numpy(self.pca.explained_variance_ratio_).float().cuda())

        print('Number of components:', self.pca.n_components_)
        print('Explained variance ratio:', self.pca.explained_variance_ratio_)
        print('Sum of explained variance ratio:', self.pca.explained_variance_ratio_.sum())

        return coeff

    def save(self, path):
        assert path.endswith('.pth'), 'Path must end with .pth'
        torch.save(self.state_dict(), path)
        print('Model saved at', path)
    
    def load(self, path):
        assert path.endswith('.pth'), 'Path must end with .pth'
        load_dict = torch.load(path, weights_only=True)
        
        self.data_mean = load_dict['data_mean']
        self.components = load_dict['components']

        self.pca.mean_ = load_dict['mean'].cpu().numpy()
        self.pca.components_ = load_dict['components'].cpu().numpy()
        self.pca.explained_variance_ = load_dict['explained_variance'].cpu().numpy()
        self.pca.explained_variance_ratio_ = load_dict['explained_variance_ratio'].cpu().numpy()

        if 'coeff_mean' in load_dict:
            self.coeff_mean = load_dict['coeff_mean']
            self.coeff_std = load_dict['coeff_std']

        print('Model loaded from', path)
    
    def encode(self, data):
        """
        data: [batch, n_params]
        """
        data_centered = data - self.data_mean
        data = data_centered @ self.components.T
        return data
    
    def decode(self, data):
        """
        data: [batch, n_components]
        """
        # data = self.pca.inverse_transform(data)
        back_to_numpy = False
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float().cuda()
            back_to_numpy = True
        
        data = data @ self.components
        data = data + self.data_mean

        if back_to_numpy:
            data = data.detach().cpu().numpy()
        return data
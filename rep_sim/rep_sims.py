import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.linalg import norm

class CKA(object):
    '''
    Calculates the linear centered kernel alignment between two sets of representations in pytorch.
    Main function runner is linear_CKA.
    >>> cka = CKA()
    >>> x, y = torch.randn(200, 768), torch.randn(200, 768)
    >>> cka.linear_cka(x, y)
    '''
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        I = torch.eye(n, device=self.device)
        H = I - torch.ones([n, n], device=self.device) / n
        return H @ K @ H 

    def linear_HSIC(self, X, Y):
        #Calculate Gram matrix.
        L_X = X @ X.T
        L_Y = Y @ Y.T
        #Center the two Gram matrices and calculate the HSIC between them i.e. the trace of their product.
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        #Numerator HSIC
        hsic = self.linear_HSIC(X, Y)
        #Denominator HSICs
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)
    
class DifferentiableRSA(object):
    '''
    Performs representational similarity analysis between two sets of representations in pytorch.
    Main function runner is rsa.
    >>> rsa = DifferentiableRSA()
    >>> x, y = torch.randn(200, 768), torch.randn(200, 768)
    >>> rsa.rsa(x, y)
    '''
    def __init__(self, device):
       self.device = device

    def compute_similarity_matrix(self, representations):
        #Normalize the representations and compute the distance between every pair of examples
        representations_norm = representations / torch.norm(representations, dim=1, keepdim=True)
        similarity_matrix = torch.mm(representations_norm, representations_norm.T)
        return similarity_matrix

    def pearson_correlation(self, x, y):
        #Flatten and compute pearson correlation
        x_flat = x.flatten()
        y_flat = y.flatten()
        x_centered = x_flat - x_flat.mean()
        y_centered = y_flat - y_flat.mean()

        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))
        return numerator / denominator
    
    def rsa(self, representations_1, representations_2):
        similarity_matrix_1 = self.compute_similarity_matrix(representations_1)
        similarity_matrix_2 = self.compute_similarity_matrix(representations_2)

        rsa_value = self.pearson_correlation(similarity_matrix_1, similarity_matrix_2)
        return rsa_value
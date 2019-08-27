import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler



class GPR_predictor:
    def __init__(self):
        super().__init__()
        self.scaler_y = None

        self.kernel = ConstantKernel() * RBF() + WhiteKernel()
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=0)

    def configure(self, x):
        len_x =len(x.shape)
        if len_x == 1:
            out = np.copy(x.reshape(-1,1))
        elif len_x == 2:
            out = np.copy(x)
        else:
            raise ValueError

        return out


    def train(self, x, y):
        x = self.configure(x)
        y = self.configure(y)

        ### scaling y
        self.scaler_y = StandardScaler().fit(y)
        self.gpr.fit(x, self.scaler_y.transform(y))

        print("Kernel: {}".format(self.gpr.kernel_))
        return 


    def predict(self, x):
        """
        Args:
            x numpy.ndarray:
                input data to be predicted
        Returns:
            pred_mu numpy.ndarray:
                predicted mean
            pred_sigma  numpy.ndarray:
                predicted std
        """
        origin_shape = x.shape
        x = self.configure(x)

        pred_mu, pred_sigma = self.gpr.predict(x, return_std=True)
        pred_mu = self.scaler_y.inverse_transform(pred_mu)
        pred_sigma = pred_sigma * self.scaler_y.scale_

        return pred_mu.reshape(origin_shape), pred_sigma.reshape(origin_shape)



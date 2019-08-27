import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler



class GPR_predictor:
    def __init__(self):
        super().__init__()
        self.scaler_y = None

        # GPモデルの構築
        self.kernel = ConstantKernel() * RBF() + WhiteKernel()
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=0)


    def train(self, x, y):
        if not len(x.shape) == 2:
            x = x.reshape(-1,1)

        if not len(y.shape) == 2:
            y = y.reshape(-1,1)

        ### scaling y
        self.scaler_y = StandardScaler().fit(y)
        self.gpr.fit(x, self.scaler_y.transform(y))

        # kernel関数のパラメータの確認
        print(self.gpr.kernel_)
        # 1.36**2 * RBF(length_scale=3.07) + WhiteKernel(noise_level=0.061)
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
        if not len(x.shape) == 2:
            x_temp = x.reshape(-1,1)


        pred_mu, pred_sigma = self.gpr.predict(x_temp, return_std=True)
        pred_mu = self.scaler_y.inverse_transform(pred_mu)
        pred_sigma = pred_sigma * self.scaler_y.scale_

        return pred_mu.reshape(x.shape), pred_sigma.reshape(x.shape)



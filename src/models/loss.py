import torch
import torch.nn as nn


class WeibullLossRMSE(nn.Module):
    """
    y_hat       : predicted RUL
    y           : true RUL
    y_days      : true age (in days)
    lambda_mod  : lambda modifier
    eta         : characteristic life
    beta        : shape parameter for weibull
    """

    def __init__(self, eps=1e-8):
        super(WeibullLossRMSE, self).__init__()
        self.eps = eps


    def forward(self, y_hat, y, y_days, lambda_mod=2.0, eta=90.0, beta=2.0):

        y_hat_days = (y_days + y) - y_hat

        # remove any "inf" values from when divided by zero
        y_hat_days = y_hat_days[torch.isfinite(y_hat_days)]

        def weibull_cdf(t, eta, beta):
            "weibull CDF function"
            return 1.0 - torch.exp(-1.0 * ((t / eta) ** beta))

        cdf = weibull_cdf(y_days, eta, beta)
        cdf_hat = weibull_cdf(y_hat_days, eta, beta)

        return lambda_mod * torch.sqrt(torch.mean(cdf_hat - cdf) ** 2 + self.eps)


class WeibullLossRMSLE(nn.Module):
    """
    y_hat       : predicted RUL
    y           : true RUL
    y_days      : true age (in days)
    lambda_mod  : lambda modifier
    eta         : characteristic life
    beta        : shape parameter for weibull
    """

    def __init__(self, eps=1e-8):
        super(WeibullLossRMSLE, self).__init__()
        self.eps = eps


    def forward(self, y_hat, y, y_days, lambda_mod=2.0, eta=90.0, beta=2.0):

        y_hat_days = (y_days + y) - y_hat

        # remove any "inf" values from when divided by zero
        y_hat_days = y_hat_days[torch.isfinite(y_hat_days)]

        def weibull_cdf(t, eta, beta):
            "weibull CDF function"
            return 1.0 - torch.exp(-1.0 * ((t / eta) ** beta))

        cdf = weibull_cdf(y_days, eta, beta)
        cdf_hat = weibull_cdf(y_hat_days, eta, beta)

        return lambda_mod * torch.sqrt(torch.mean(torch.log(cdf_hat + 1) - torch.log(cdf+1)) ** 2 + self.eps)


class WeibullLossMSE(nn.Module):
    """
    y_hat       : predicted RUL
    y           : true RUL
    y_days      : true age (in days)
    lambda_mod  : lambda modifier
    eta         : characteristic life
    beta        : shape parameter for weibull
    """

    def __init__(self):
        super(WeibullLossMSE, self).__init__()


    def forward(self, y_hat, y, y_days, lambda_mod=2.0, eta=90.0, beta=2.0):

        y_hat_days = (y_days + y) - y_hat

        # remove any "inf" values from when divided by zero
        y_hat_days = y_hat_days[torch.isfinite(y_hat_days)]

        def weibull_cdf(t, eta, beta):
            "weibull CDF function"
            return 1.0 - torch.exp(-1.0 * ((t / eta) ** beta))

        cdf = weibull_cdf(y_days, eta, beta)
        cdf_hat = weibull_cdf(y_hat_days, eta, beta)

        return lambda_mod * torch.mean((cdf_hat - cdf) ** 2)


class RMSELoss(nn.Module):
    # https://discuss.pytorch.org/t/rmse-loss-function/16540/4

    def __init__(self, eps=1e-8):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y) + self.eps)


class RMSLELoss(nn.Module):
    # https://discuss.pytorch.org/t/rmse-loss-function/16540/4

    def __init__(self, eps=1e-8):
        super(RMSLELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(torch.log(y_hat + 1), torch.log(y + 1)) + self.eps)


class MAPELoss(nn.Module):

    def __init__(self, eps=1e-8):
        super(MAPELoss, self).__init__()
        self.eps = eps

    def forward(self, y_hat, y):

        return torch.mean(torch.abs(y - y_hat) / torch.abs(y + self.eps)) * 100
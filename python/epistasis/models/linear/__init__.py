"""Linear epistasis models."""

from epistasis.models.linear.elastic_net import EpistasisElasticNet
from epistasis.models.linear.lasso import EpistasisLasso
from epistasis.models.linear.ordinary import EpistasisLinearRegression
from epistasis.models.linear.ridge import EpistasisRidge

__all__ = [
    "EpistasisElasticNet",
    "EpistasisLasso",
    "EpistasisLinearRegression",
    "EpistasisRidge",
]

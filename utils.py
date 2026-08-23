import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import torch
import torch.nn.functional as F
import os
import logging
from datetime import datetime
import time
import random
import torch.nn as nn

def getlogger(path=None, level=logging.INFO):
    """
        Get logger for logging.
    """
    TIMESTAMP = "{0:%m-%d-%H-%M}".format(datetime.now())
    logger = logging.getLogger()
    logger.setLevel(level)
    rq = time.strftime(TIMESTAMP)
    log_name = os.path.join(path, rq + '.log')
    logfile = log_name
    print("log file: {}".format(logfile))
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    """
        The logistic function used for fitting.
    """
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    """
        Fit a logistic function to the data using non-linear least squares optimization.
    """
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def performance_fit(y_label, y_output,if_logistic=True):
    """
        Calculate Pearson correlation coefficient, Spearman rank correlation coefficient,
        Kendall rank correlation coefficient and Root Mean Square Error between two lists of numbers.
    """
    if not if_logistic:
        y_output_logistic = y_output
    else:
        y_output_logistic = fit_function(y_label, y_output)
    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_output_logistic-y_label) ** 2).mean())

    return PLCC, SRCC, KRCC, RMSE


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

class PLCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(PLCCLoss, self).__init__()
        self.eps = eps

    def forward(self, pred_scores, gt_scores):
        """
        The loss defined in the paper is the negative PLCC.
        Args:
            pred_scores: predicted scores of shape (batch_size,)
            gt_scores: ground truth scores of shape (batch_size,)
        """
        # flatten the scores to 1D
        pred_scores = pred_scores.view(-1)
        gt_scores = gt_scores.view(-1)
        
        # calculate the mean of the scores
        pred_mean = pred_scores.mean()
        gt_mean = gt_scores.mean()
        
        # normalize the scores by subtracting the mean
        pred_centered = pred_scores - pred_mean
        gt_centered = gt_scores - gt_mean
        
        # calculate the numerator and denominator of the PLCC formula
        numerator = (pred_centered * gt_centered).sum()
        pred_std = torch.sqrt((pred_centered ** 2).sum() + self.eps)
        gt_std = torch.sqrt((gt_centered ** 2).sum() + self.eps)
     
        plcc = numerator / (pred_std * gt_std + self.eps)
        
        return 1-plcc  # 最大化相关性


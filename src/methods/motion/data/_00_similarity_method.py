import cv2
import numpy as np

from scipy.signal import convolve2d
from scipy.stats import spearmanr, pearsonr


from methods.motion.data._00_data_preprocess import get_normalized_fft


def get_euclidean_score(frames, signal, alpha=1):
    transposed = np.transpose(frames, (1, 2, 3, 0))
    diff1 = transposed - signal
    diff2 = -transposed - signal
    distance1 = np.sqrt(np.sum(np.power(diff1, 10), axis=-1))
    distance2 = np.sqrt(np.sum(np.power(diff2, 10), axis=-1))
    channel_wise_mean1 = np.mean(distance1, axis=-1)
    channel_wise_mean2 = np.mean(distance2, axis=-1)
    distance = np.minimum(channel_wise_mean1, channel_wise_mean2)
    score = np.exp(-distance * alpha)
    return score


def get_dot_score(frames, signal):
    transposed = np.transpose(frames, (1, 2, 3, 0))
    dot = np.sum(transposed * signal, axis=-1)
    abs_dot = np.fabs(dot)
    channel_wise_mean = np.mean(abs_dot, axis=-1)
    score = 1 - np.exp(-channel_wise_mean)
    return score


def get_cosine_score(frames, signal):
    transposed = np.transpose(frames, (1, 2, 3, 0))
    dot1 = np.sum(transposed * signal, axis=-1)
    dot2 = np.sum(-transposed * signal, axis=-1)
    signal_size = np.sqrt(np.sum(np.square(signal)))
    frame_size = np.sqrt(np.sum(np.square(transposed), axis=-1))
    div = frame_size.copy()
    div[div == 0] = 1
    cos1 = dot1 / (signal_size * div)
    cos2 = dot2 / (signal_size * div)
    cos1[frame_size == 0] = 0
    cos2[frame_size == 0] = 0
    cos1 = np.mean(cos1, axis=-1)
    cos2 = np.mean(cos2, axis=-1)
    cos = np.maximum(cos1, cos2)
    return cos


def get_snr_score(frames, signal):
    trans = frames.transpose((1, 2, 3, 0))
    diff1 = trans - signal
    diff2 = -trans - signal

    ps = np.square(signal).sum()
    pn1 = np.square(diff1).sum(axis=-1)
    pn2 = np.square(diff2).sum(axis=-1)

    pn = np.minimum(pn1, pn2)
    pn = pn.mean(axis=-1)

    snr = 10 * np.log(ps / pn)
    thres = 0
    snr[snr <= thres] = 0
    snr[snr > thres] = 1
    return snr


def get_pearson_correlation_score(frames, signal):
    l, h, w = frames.shape[:3]
    c = frames.shape[-1] if len(frames.shape) == 4 else 1

    coefs = []
    for i in range(h):
        signals = frames[:, i, ...]
        signals = signals.reshape((l, -1)).T

        mean = np.mean((signals == 0).astype(np.int), axis=1)
        signals[mean==1.0, -1] = 1

        coef = np.corrcoef(signal, signals)
        coefs.append(np.fabs(coef[0][1:]))
    coef = np.array(coefs).reshape((h, w, c))
    if coef.shape[-1] == 3:
        max_coef = np.maximum(np.maximum(coef[..., 0], coef[..., 1]), coef[..., 2])
    else:
        max_coef = coef[..., 0]
    return max_coef


def get_spearman_correlation_score(frames, signal):
    l, h, w = frames.shape[:3]
    c = frames.shape[-1] if len(frames.shape) == 4 else 1

    coefs = []
    for i in range(h):
        signals = frames[:, i, ...]
        signals = signals.reshape((l, -1)).T

        coef = spearmanr(signal, signals, axis=1, nan_policy='omit')
        if coef[0] is np.nan:
            coef = np.zeros((1, 1, 241))
            coef[:] = np.nan
        coefs.append(np.fabs(coef[0][0][1:]))
    coef = np.array(coefs).reshape((h, w, c))
    if coef.shape[-1] == 3:
        max_coef = np.maximum(np.maximum(coef[..., 0], coef[..., 1]), coef[..., 2])
    else:
        max_coef = coef[..., 0]
    return max_coef


def old_label_method(xs, ys, thres=0.02):
    fft_xs = get_normalized_fft(xs, axis=0)
    fft_ys = get_normalized_fft(ys, axis=0)

    score = get_euclidean_score(xs, ys)
    score *= get_dot_score(xs, ys)
    score *= get_cosine_score(xs, ys)
    score *= get_euclidean_score(fft_xs, fft_ys)
    score *= get_dot_score(fft_xs, fft_ys)
    score *= get_cosine_score(fft_xs, fft_ys)

    if thres is not None:
        score[score > thres] = 1
        score[score <= thres] = 0

    if len(score.shape) == 2:
        h, w = score.shape
        score = score.reshape((h, w, 1))

    return score
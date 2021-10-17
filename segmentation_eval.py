import numpy as np


def computeMetrics(confusion):
    """
    Compute evaluation metrics given a confusion matrix.
    :param confusion: any confusion matrix
    :return: tuple (miou, fwiou, macc, pacc, ious, maccs)
    """
    # Init
    labelCount = confusion.shape[0]
    ious = np.zeros((labelCount))
    maccs = np.zeros((labelCount))
    ious[:] = np.NAN
    maccs[:] = np.NAN

    # Get true positives, positive predictions and positive ground-truth
    total = confusion.sum()
    if total <= 0:
        raise Exception('Error: Confusion matrix is empty!')
    tp = np.diagonal(confusion)
    posPred = confusion.sum(axis=0)
    posGt = confusion.sum(axis=1)

    # Check which classes have elements
    valid = posGt > 0
    iousValid = np.logical_and(valid, posGt + posPred - tp > 0)

    # Compute per-class results and frequencies
    ious[iousValid] = np.divide(tp[iousValid], posGt[iousValid] + posPred[iousValid] - tp[iousValid])
    maccs[valid] = np.divide(tp[valid], posGt[valid])
    freqs = np.divide(posGt, total)

    # Compute evaluation metrics
    miou = np.mean(ious[iousValid])
    fwiou = np.sum(np.multiply(ious[iousValid], freqs[iousValid]))
    macc = np.mean(maccs[valid])
    pacc = tp.sum() / total

    return miou, fwiou, macc, pacc, ious, maccs

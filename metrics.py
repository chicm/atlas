from sklearn import metrics
import torch
import numpy as np
import torch.nn.functional as F

def accuracy(logits, target):
    '''
    logits: N,C
    target: N,C
    '''
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    results = []
    target = target.byte()
    #pred = pred.topk(3, )
    for t in thresholds:
        pred = (torch.sigmoid(logits) > t).byte()
        corrects = (pred.eq(target) * target).sum().item()
        incorrects = (pred.ne(target) * (target.eq(0))).sum().item()
        results.append((corrects, incorrects, round(corrects/(incorrects+1),3), t, target.sum().item()))
    return results

def accuracy_th(logits, target, thresholds):
    '''
    logits: N,C
    target: N,C
    '''
    #thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    results = []
    target = target.byte()
    #pred = pred.topk(3, )
    #for t in thresholds:
    pred = (torch.sigmoid(logits) > thresholds).byte()
    corrects = (pred.eq(target) * target).sum().item()
    incorrects = (pred.ne(target) * (target.eq(0))).sum().item()
    results.append((corrects, incorrects, round(corrects/(incorrects+1),3), 'optimized'))
    
    return results

def find_fix_threshold(preds, targets):
    #print('>>>', logits.size(), targets.size())
    assert preds.size() == targets.size()

    best_t = 0.01
    best_score = 0.
    for t in range(1, 100):
        cur_th = t/100.
        preds_t = (preds > cur_th).float()
        score = f2_score(targets, preds_t)
        if score > best_score:
            best_score = score
            best_t = cur_th
    #print(thresholds)
    return best_t


def find_threshold(logits, targets):
    #print('>>>', logits.size(), targets.size())
    N_CLASSESS = logits.size(1)
    assert logits.size() == targets.size()

    thresholds = [0.15]*N_CLASSESS
    outputs = torch.sigmoid(logits)
    for i in range(N_CLASSESS):
        best_t = 0.15
        best_score = f2_score(targets, outputs, threshold=torch.Tensor(thresholds).cuda())
        for t in range(99):
            cur_th = t/100.+0.001
            thresholds[i] = cur_th
            score = f2_score(targets, outputs, threshold=torch.Tensor(thresholds).cuda())
            if score > best_score:
                best_score = score
                best_t = cur_th
        thresholds[i] = best_t
    #print(thresholds)
    return thresholds

def topk_accuracy(output, label, topk=(1,100)):
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum().item()
        res.append(correct_k)
    return res

def f2_scores(logits, target):
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    preds = torch.sigmoid(logits)
    results = []
    for t in thresholds:
        preds_t = (preds > t).float()
        score = round(f2_score(target, preds_t).item(), 4)
        results.append((t, score))
    return results

def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, 2)


def fbeta_score(y_true, y_pred, beta, eps=1e-9):
    beta2 = beta**2

    #y_pred = torch.ge(y_pred.float(), threshold).float()
    y_pred = y_pred.float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))

def test_f2():
    y_pred = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

    y_true = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

    py_pred = torch.from_numpy(y_pred)
    py_true = torch.from_numpy(y_true)

    fbeta_pytorch = f2_score(py_true, py_pred)
    #fbeta_sklearn = metrics.fbeta_score(y_true, y_pred, 2, average='samples')
    #print('Scores are {:.3f} (sklearn) and {:.3f} (pytorch)'.format(fbeta_sklearn, fbeta_pytorch))
    print(fbeta_pytorch.cuda().item())
    print('Scores are {:.3f}'.format(fbeta_pytorch))

def test_f2_2():
    y_pred = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]])

    y_true = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]])

    py_pred = torch.from_numpy(y_pred)
    py_true = torch.from_numpy(y_true)

    fbeta_pytorch = f2_score(py_true, py_pred)
    #fbeta_sklearn = metrics.fbeta_score(y_true, y_pred, 2, average='samples')
    #print('Scores are {:.3f} (sklearn) and {:.3f} (pytorch)'.format(fbeta_sklearn, fbeta_pytorch))
    print(fbeta_pytorch.cuda().item())
    print('Scores are {:.3f}'.format(fbeta_pytorch))

if __name__ == '__main__':
    test_f2_2()

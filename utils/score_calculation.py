"""
Post-hoc OOD scoring functions: ODIN, Mahalanobis.

These are used as baselines or complementary scoring methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)


# ---------------------------------------------------------------------------
# ODIN scoring
# ---------------------------------------------------------------------------

def get_ood_scores_odin(loader, net, bs, ood_num_examples, T, noise, in_dist=False):
    """Compute ODIN scores for a data loader.

    Returns:
        If in_dist: (all_scores, right_scores, wrong_scores)
        Else: ood_scores array of length ood_num_examples
    """
    _score, _right_score, _wrong_score = [], [], []
    net.eval()

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= ood_num_examples // bs and not in_dist:
            break

        data = data.cuda()
        data.requires_grad_(True)
        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODIN(data, output, net, T, noise)
        _score.append(-np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right = preds == targets
            _right_score.append(-np.max(smax[right], axis=1))
            _wrong_score.append(-np.max(smax[~right], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    return concat(_score)[:ood_num_examples].copy()


def ODIN(inputs, outputs, model, temper, noise_magnitude):
    """Compute ODIN-perturbed softmax scores."""
    criterion = nn.CrossEntropyLoss()
    max_idx = np.argmax(outputs.data.cpu().numpy(), axis=1)
    outputs = outputs / temper
    labels = torch.LongTensor(max_idx).cuda()
    loss = criterion(outputs, labels)
    loss.backward()

    gradient = torch.ge(inputs.grad.data, 0).float()
    gradient = (gradient - 0.5) * 2

    # Normalise per channel
    gradient[:, 0] /= 63.0 / 255.0
    gradient[:, 1] /= 62.1 / 255.0
    gradient[:, 2] /= 66.7 / 255.0

    temp_inputs = inputs.data - noise_magnitude * gradient
    outputs = model(temp_inputs)
    outputs = outputs / temper

    out_np = outputs.data.cpu().numpy()
    out_np = out_np - np.max(out_np, axis=1, keepdims=True)
    out_np = np.exp(out_np) / np.sum(np.exp(out_np), axis=1, keepdims=True)
    return out_np


# ---------------------------------------------------------------------------
# Mahalanobis scoring
# ---------------------------------------------------------------------------

def get_Mahalanobis_score(model, test_loader, num_classes, sample_mean,
                          precision, layer_index, magnitude, num_batches,
                          in_dist=False):
    """Compute Mahalanobis confidence scores for OOD detection."""
    model.eval()
    scores = []

    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx >= num_batches and not in_dist:
            break

        data = data.cuda()
        data.requires_grad_(True)

        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # Compute Gaussian log-likelihood per class
        gaussian_score = None
        for i in range(num_classes):
            mean_i = sample_mean[layer_index][i]
            diff = out_features.data - mean_i
            term = -0.5 * torch.mm(torch.mm(diff, precision[layer_index]), diff.t()).diag()
            if gaussian_score is None:
                gaussian_score = term.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term.view(-1, 1)), 1)

        # Input preprocessing
        pred = gaussian_score.max(1)[1]
        batch_mean = sample_mean[layer_index].index_select(0, pred)
        diff = out_features - batch_mean
        pure_gau = -0.5 * torch.mm(torch.mm(diff, precision[layer_index]), diff.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(data.grad.data, 0).float()
        gradient = (gradient - 0.5) * 2
        gradient[:, 0] /= 63.0 / 255.0
        gradient[:, 1] /= 62.1 / 255.0
        gradient[:, 2] /= 66.7 / 255.0

        temp_inputs = data.data - magnitude * gradient
        with torch.no_grad():
            noise_features = model.intermediate_forward(temp_inputs, layer_index)
        noise_features = noise_features.view(noise_features.size(0), noise_features.size(1), -1)
        noise_features = torch.mean(noise_features, 2)

        noise_gaussian = None
        for i in range(num_classes):
            mean_i = sample_mean[layer_index][i]
            diff = noise_features.data - mean_i
            term = -0.5 * torch.mm(torch.mm(diff, precision[layer_index]), diff.t()).diag()
            if noise_gaussian is None:
                noise_gaussian = term.view(-1, 1)
            else:
                noise_gaussian = torch.cat((noise_gaussian, term.view(-1, 1)), 1)

        noise_score, _ = torch.max(noise_gaussian, dim=1)
        scores.extend(-noise_score.cpu().numpy())

    return np.asarray(scores, dtype=np.float32)


def sample_estimator(model, num_classes, feature_list, train_loader):
    """Compute class-conditional mean and precision (inverse covariance)
    for the Mahalanobis detector.

    Returns:
        (sample_class_mean, precision): lists indexed by layer.
    """
    import sklearn.covariance

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)

    num_sample_per_class = np.zeros(num_classes)
    list_features = [[0] * num_classes for _ in range(num_output)]

    for data, target in train_loader:
        total += data.size(0)
        with torch.no_grad():
            data = data.cuda()
            output, out_features = model.feature_list(data)

        for i in range(num_output):
            feat = out_features[i]
            feat = feat.view(feat.size(0), feat.size(1), -1)
            feat = torch.mean(feat.data, 2)
            out_features[i] = feat

        pred = output.data.max(1)[1]
        correct += pred.eq(target.cuda()).cpu().sum()

        for j in range(data.size(0)):
            label = target[j]
            for i in range(num_output):
                vec = out_features[i][j].view(1, -1)
                if num_sample_per_class[label] == 0:
                    list_features[i][label] = vec
                else:
                    list_features[i][label] = torch.cat((list_features[i][label], vec), 0)
            num_sample_per_class[label] += 1

    # Compute class means
    sample_class_mean = []
    for i, num_feat in enumerate(feature_list):
        temp = torch.Tensor(num_classes, int(num_feat)).cuda()
        for j in range(num_classes):
            temp[j] = torch.mean(list_features[i][j], 0)
        sample_class_mean.append(temp)

    # Compute precision matrices
    precision = []
    for k in range(num_output):
        X = None
        for i in range(num_classes):
            diff = list_features[k][i] - sample_class_mean[k][i]
            X = diff if X is None else torch.cat((X, diff), 0)
        group_lasso.fit(X.cpu().numpy())
        temp = torch.from_numpy(group_lasso.precision_).float().cuda()
        precision.append(temp)

    print(f"\n Training Accuracy: ({100.0 * correct / total:.2f}%)\n")
    return sample_class_mean, precision

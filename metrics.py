import torch
import torch.nn.functional as F
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, roc_auc_score
)

def _probs_and_losses(output, labels, output_is_prob=False):
    """Probabilidades y loss por muestra a partir de la salida de `predict`."""
    output = output.float()
    if output_is_prob:                       # ADAPTAR: True si `output` ya suma 1 por fila
        probs = output
        losses = -torch.log(probs.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12))
    else:                                    # `output` son logits
        probs = torch.softmax(output, dim=1)
        losses = F.cross_entropy(output, labels, reduction="none")
    return probs, losses

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    probs, preds, labels, weights, losses = [], [], [], [], []
    for x, y in loader:                      # adaptar si el batch trae más elementos
        x, y = x.to(device), y.to(device)
        logits = model(x)                    # salida del ensemble/dispatcher
        probs.append(torch.softmax(logits, dim=1).cpu())
        preds.append(logits.argmax(1).cpu())
        labels.append(y.cpu())
        weights.append(torch.ones(len(y)))
        losses.append(F.cross_entropy(logits, y, reduction="none").cpu())
    model.train()
    return {
        "probs": torch.cat(probs).numpy(), "preds": torch.cat(preds).numpy(),
        "labels": torch.cat(labels).numpy(), "weights": torch.cat(weights).numpy(),
        "losses": torch.cat(losses).numpy(),
    }

def balanced_accuracy(results):
    return balanced_accuracy_score(
        results["labels"], results["preds"], sample_weight=results["weights"]
    )


def macro_f1(results):
    return f1_score(
        results["labels"],
        results["preds"],
        average="macro",
        sample_weight=results["weights"],
    )

def binary_f1(results, positive_class, num_classes=None):
    """F1 binario con `positive_class` como clase positiva.

    Devuelve nan si el problema no es binario, en vez de fallar.
    """
    if num_classes is not None and num_classes != 2:
        return float("nan")

    labels = np.asarray(results["labels"])
    preds = np.asarray(results["preds"])
    weights = np.asarray(results["weights"])

    # Red de seguridad: si aparece cualquier clase fuera de {0, 1}
    # (en labels o en preds), average="binary" lanzaría ValueError.
    if not np.isin(np.union1d(labels, preds), [0, 1]).all():
        return float("nan")

    return f1_score(
        labels,
        preds,
        average="binary",
        pos_label=positive_class,
        sample_weight=weights,
        zero_division=0,
    )


def make_binary_f1(positive_class, num_classes=None):
    def metric_fn(results):
        return binary_f1(results, positive_class, num_classes)
    return metric_fn

def _sensitivity_specificity_per_class(results):
    """Calcula sensitivity y specificity one-vs-rest para cada clase."""
    labels = results["labels"]
    preds = results["preds"]
    weights = results["weights"]

    out = {}

    for c in np.unique(labels):
        y_true_bin = (labels == c).astype(int)
        y_pred_bin = (preds == c).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true_bin,
            y_pred_bin,
            sample_weight=weights,
            labels=[0, 1],
        ).ravel()

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else float("nan")
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else float("nan")
        )

        out[int(c)] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    return out

def mean_loss(results):
    return float(np.average(results["losses"], weights=results["weights"]))

def auroc_(results):
    labels, probs, w = results["labels"], results["probs"], results["weights"]

    if len(np.unique(labels)) < 2:
        # AUROC no está definido si el dominio solo tiene una clase presente
        return float("nan")

    if probs.ndim == 1:
        # Red con salida escalar (p.size(1) == 1 en get_predictions)
        return roc_auc_score(labels, probs, sample_weight=w)
    elif probs.shape[1] == 2:
        # Softmax de 2 columnas pero problema binario -> solo prob. de la clase positiva
        return roc_auc_score(labels, probs[:, 1], sample_weight=w)
    else:
        # Multiclase real (3+ clases)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / row_sums

        return roc_auc_score(
            labels,
            probs,
            labels=np.arange(probs.shape[1]),  # explícito: columna i = clase i
            multi_class="ovr",
            average="macro",
            sample_weight=w,
        )


def auroc(results):
    from torchmetrics.functional.classification import multiclass_auroc
    labels, probs, w = results["labels"], results["probs"], results["weights"]

    if len(np.unique(labels)) < 2:
        # AUROC no está definido si el dominio solo tiene una clase presente
        return float("nan")

    if probs.ndim == 1:
        # Red con salida escalar (p.size(1) == 1 en get_predictions)
        return roc_auc_score(labels, probs, sample_weight=w)
    elif probs.shape[1] == 2:
        # Softmax de 2 columnas pero problema binario -> solo prob. de la clase positiva
        return roc_auc_score(labels, probs[:, 1], sample_weight=w)
    else:
        # Multiclase real (3+ clases): macro-AUROC one-vs-rest robusto
        labels = np.asarray(labels).astype(np.int64).ravel()
        probs = np.asarray(probs, dtype=np.float64)   # float64 para evitar problemas de precisión
        if w is not None:
            w = np.asarray(w, dtype=np.float64).ravel()

        # Descartar filas con NaN/inf (una sola fila rompería toda la métrica)
        finite = np.isfinite(probs).all(axis=1)
        if not finite.all():
            print(f"[auroc] descartando {(~finite).sum()} filas con NaN/inf")
            labels, probs = labels[finite], probs[finite]
            if w is not None:
                w = w[finite]

        # Clases evaluables: con positivos y negativos en este dominio
        C = probs.shape[1]
        counts = np.bincount(labels, minlength=C)
        valid = (counts > 0) & (counts < len(labels))
        if not valid.any():
            return float("nan")

        uniform_w = w is None or np.allclose(w, w.flat[0])
        if uniform_w:
            # Sin pesos relevantes: torchmetrics (no exige que las filas sumen 1)
            per_class = multiclass_auroc(
                torch.from_numpy(probs),
                torch.from_numpy(labels),
                num_classes=C,
                average=None,
            )
            return float(per_class[torch.from_numpy(valid)].mean().item())
        else:
            # Pesos no uniformes: OvR manual con sklearn, que sí soporta sample_weight
            aucs = [
                roc_auc_score((labels == c).astype(int), probs[:, c], sample_weight=w)
                for c in np.flatnonzero(valid)
            ]
            return float(np.mean(aucs))
        

def _accuracy(results):
    return accuracy_score(
        results["labels"], results["preds"], sample_weight=results["weights"]
    )

def sensitivity(results):
    """Sensitivity one-vs-rest para cada clase."""
    per_class = _sensitivity_specificity_per_class(results)

    return {
        c: values["sensitivity"]
        for c, values in per_class.items()
    }

def specificity(results):
    """Specificity one-vs-rest para cada clase."""
    per_class = _sensitivity_specificity_per_class(results)

    return {
        c: values["specificity"]
        for c, values in per_class.items()
    }

def macro_sensitivity(results):
    """Media macro de sensitivity entre clases."""
    values = sensitivity(results).values()
    return float(np.nanmean(list(values)))

def macro_specificity(results):
    """Media macro de specificity entre clases."""
    values = specificity(results).values()
    return float(np.nanmean(list(values)))


def compute_split_metrics(preds_by_split, extra_metrics):
    results = {}
    for name, r in preds_by_split.items():
        results[f"{name}_acc"] = _accuracy(r)
        results[f"{name}_loss"] = mean_loss(r)
        for m_name, m_fn in extra_metrics.items():
            results[f"{name}_{m_name}"] = m_fn(r)
    return results


def write_outputs(path, preds_by_split, step):
    record = {n: {k: np.asarray(v).tolist() for k, v in r.items()}
              for n, r in preds_by_split.items()}
    record["step"] = step
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

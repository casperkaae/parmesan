import numpy as np
import theano.tensor as T


def log_sum_exp(A, axis=None, sum_op=T.sum):
    """Computes `log(exp(A).sum(axis=axis))` avoiding numerical issues using the log-sum-exp trick.

    Direct calculation of :math:`\log \sum_i \exp A_i` can result in underflow or overflow numerical 
    issues. Big positive values can cause overflow :math:`\exp A_i = \inf`, and big negative values 
    can cause underflow :math:`\exp A_i = 0`. The latter can eventually cause the sum to go to zero 
    and finally resulting in :math:`\log 0 = -\inf`.

    The log-sum-exp trick avoids these issues by using the identity,

    .. math::
        \log \sum_i \exp A_i = \log \sum_i \exp(A_i - c) + c, \text{using},  \\
        c = \max A.

    This avoids overflow, and while underflow can still happen for individual elements it avoids 
    the sum being zero.
     
    Parameters
    ----------
    A : Theano tensor
        Tensor of which we wish to compute the log-sum-exp.
    axis : int, tuple, list, None
        Axis or axes to sum over; None (default) sums over all axes.
    sum_op : function
        Summing function to apply; default is T.sum, but can also be T.mean for log-mean-exp.
		
    Returns
    -------
    Theano tensor
        The log-sum-exp of `A`, dimensions over which is summed will be dropped.
    """
    A_max = T.max(A, axis=axis, keepdims=True)
    B = T.log(sum_op(T.exp(A - A_max), axis=axis, keepdims=True)) + A_max

    if axis is None:
        return B.dimshuffle(())  # collapse to scalar
    else:
        if not hasattr(axis, '__iter__'): axis = [axis]
        return B.dimshuffle([d for d in range(B.ndim) if d not in axis])  # drop summed axes

def log_mean_exp(A, axis=None):
    """Computes `log(exp(A).mean(axis=axis))` avoiding numerical issues using the log-sum-exp trick.

    See also
    --------
    log_sum_exp
    """
    return log_sum_exp(A, axis, sum_op=T.mean)


class ConfusionMatrix:
    """
       Simple confusion matrix class
       row is the true class, column is the predicted class
    """
    def __init__(self, n_classes, class_names=None):
        self.n_classes = n_classes
        if class_names is None:
            self.class_names = map(str, range(n_classes))
        else:
            self.class_names = class_names

        # find max class_name and pad
        max_len = max(map(len, self.class_names))
        self.max_len = max_len
        for idx, name in enumerate(self.class_names):
            if len(self.class_names) < max_len:
                self.class_names[idx] = name + " "*(max_len-len(name))

        self.mat = np.zeros((n_classes, n_classes), dtype='int')

    def __str__(self):
        # calucate row and column sums
        col_sum = np.sum(self.mat, axis=1)
        row_sum = np.sum(self.mat, axis=0)

        s = []

        mat_str = self.mat.__str__()
        mat_str = mat_str.replace('[', '').replace(']', '').split('\n')

        for idx, row in enumerate(mat_str):
            if idx == 0:
                pad = " "
            else:
                pad = ""
            class_name = self.class_names[idx]
            class_name = " " + class_name + " |"
            row_str = class_name + pad + row
            row_str += " |" + str(col_sum[idx])
            s.append(row_str)

        row_sum = [(self.max_len+4)*" "+" ".join(map(str, row_sum))]
        hline = [(1+self.max_len)*" "+"-"*len(row_sum[0])]

        s = hline + s + hline + row_sum

        # add linebreaks
        s_out = [line+'\n' for line in s]

        return "".join(s_out)

    def batchadd(self, y_pred, y_true):
        assert y_true.shape == y_pred.shape
        assert len(y_true) == len(y_pred)
        assert max(y_true) < self.n_classes
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        for i in range(len(y_true)):
                self.mat[y_true[i], y_pred[i]] += 1

    def batchaddmask(self, y_true, y_pred, mask):
        assert y_true.shape == y_pred.shape
        assert y_true.shape == mask.shape
        assert mask.dtype == np.bool, \
            "performance will be wrong if this is ints"
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        self.batchAdd(y_true_masked, y_pred_masked)

    def zero(self):
        self.mat.fill(0)


    def geterrors(self):
        """
        Calculate differetn error types

        Returns
        -------
        vectors of true postives (tp) false negatives (fn),
        false positives (fp) and true negatives (tn)
        pos 0 is first class, pos 1 is second class etc.
        """
        tp = np.asarray(np.diag(self.mat).flatten(), dtype='float')
        fn = np.asarray(np.sum(self.mat, axis=1).flatten(), dtype='float') - tp
        fp = np.asarray(np.sum(self.mat, axis=0).flatten(), dtype='float') - tp
        tn = np.asarray(np.sum(self.mat)*np.ones(self.n_classes).flatten(),
                        dtype='float') - tp - fn - fp
        return tp, fn, fp, tn

    def accuracy(self):
        """
        Calculates global accuracy
        :return: accuracy
        :example: >>> conf = ConfusionMatrix(3)
                  >>> conf.batchAdd([0,0,1],[0,0,2])
                  >>> print conf.accuracy()
        """
        tp, _, _, _ = self.geterrors()
        n_samples = np.sum(self.mat)
        return np.sum(tp) / n_samples

    def sensitivity(self):
        tp, tn, fp, fn = self.geterrors()
        res = tp / (tp + fn)
        res = res[~np.isnan(res)]
        return res

    def specificity(self):
        tp, tn, fp, fn = self.geterrors()
        res = tn / (tn + fp)
        res = res[~np.isnan(res)]
        return res

    def positivepredictivevalue(self):
        tp, tn, fp, fn = self.geterrors()
        res = tp / (tp + fp)
        res = res[~np.isnan(res)]
        return res

    def negativepredictivevalue(self):
        tp, tn, fp, fn = self.geterrors()
        res = tn / (tn + fn)
        res = res[~np.isnan(res)]
        return res

    def falsepositiverate(self):
        tp, tn, fp, fn = self.geterrors()
        res = fp / (fp + tn)
        res = res[~np.isnan(res)]
        return res

    def falsediscoveryrate(self):
        tp, tn, fp, fn = self.geterrors()
        res = fp / (tp + fp)
        res = res[~np.isnan(res)]
        return res

    def F1(self):
        tp, tn, fp, fn = self.geterrors()
        res = (2*tp) / (2*tp + fp + fn)
        res = res[~np.isnan(res)]
        return res

    def matthewscorrelation(self):
        tp, tn, fp, fn = self.geterrors()
        numerator = tp*tn - fp*fn
        denominator = np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
        res = numerator / denominator
        res = res[~np.isnan(res)]
        return res


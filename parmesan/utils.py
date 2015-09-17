import numpy as np


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

    def batchadd(self, y_true, y_pred):
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
        tp, _, _, _ = self.getErrors()
        n_samples = np.sum(self.mat)
        return np.sum(tp) / n_samples

    def sensitivity(self):
        tp, tn, fp, fn = self.getErrors()
        res = tp / (tp + fn)
        res = res[~np.isnan(res)]
        return res

    def specificity(self):
        tp, tn, fp, fn = self.getErrors()
        res = tn / (tn + fp)
        res = res[~np.isnan(res)]
        return res

    def positivepredictivevalue(self):
        tp, tn, fp, fn = self.getErrors()
        res = tp / (tp + fp)
        res = res[~np.isnan(res)]
        return res

    def negativepredictivevalue(self):
        tp, tn, fp, fn = self.getErrors()
        res = tn / (tn + fn)
        res = res[~np.isnan(res)]
        return res

    def falsepositiverate(self):
        tp, tn, fp, fn = self.getErrors()
        res = fp / (fp + tn)
        res = res[~np.isnan(res)]
        return res

    def falsediscoveryrate(self):
        tp, tn, fp, fn = self.getErrors()
        res = fp / (tp + fp)
        res = res[~np.isnan(res)]
        return res

    def F1(self):
        tp, tn, fp, fn = self.getErrors()
        res = (2*tp) / (2*tp + fp + fn)
        res = res[~np.isnan(res)]
        return res

    def matthewscorrelation(self):
        tp, tn, fp, fn = self.getErrors()
        numerator = tp*tn - fp*fn
        denominator = np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
        res = numerator / denominator
        res = res[~np.isnan(res)]
        return res


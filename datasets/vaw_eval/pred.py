import numpy as np

if __name__ == '__main__':

    predictions = np.load('pred.npy') #(31819, 620)
    annotations = np.load('gt_label.npy') #(31819, 620)
    import pdb; pdb.set_trace()
import json
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == '__main__':

    ap_path = 'ap.json'
    non_rare_ap_path = 'non_rare_ap.json'
    rare_ap_path = 'rare_ap.json'
    max_reacll_path = 'max_recall.json'
    
    # with open(ap_path, 'r') as f:
    #     ap = json.load(f)

    # ap_list = []
    # for i,key in enumerate(ap.keys()):
    #     value = ap[key]
    #     ap_list.append({'index': key, 'ap':value})
    # df = pd.DataFrame(ap_list)
    # top_20 = df.sort_values(by=['ap'],ascending=False)[:20]
    #import pdb; pdb.set_trace()


    with open(ap_path, 'r') as f:
        ap = json.load(f)

    ap_list = []
    for key in ap.keys():
        value = ap[key]
        ap_list.append({'class': key, 'ap':value})

    df = pd.DataFrame(ap_list)
    top_20 = df.sort_values(by=['ap'],ascending=False)[:20]
    import pdb; pdb.set_trace()
    
    with open(non_rare_ap_path, 'r') as f:
        non_rare_ap = json.load(f)

    ap_list2 = []
    for i,key in enumerate(non_rare_ap.keys()):
        value = non_rare_ap[key]
        ap_list2.append({'index': key, 'ap':value})

    df2 = pd.DataFrame(ap_list2)
    top_20_2 = df2.sort_values(by=['ap'],ascending=False)[:20]
    #import pdb; pdb.set_trace()
    with open(rare_ap_path, 'r') as f:
        rare_ap = json.load(f)

    with open(max_reacll_path, 'r') as f:
        max_reacall = json.load(f)

    

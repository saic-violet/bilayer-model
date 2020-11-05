import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch

pascal_graph = {0:[0],
                1:[1, 2],
                2:[1, 2, 3, 5],
                3:[2, 3, 4],
                4:[3, 4],
                5:[2, 5, 6],
                6:[5, 6]}

cihp_graph = {0: [],
              1: [2, 13],
              2: [1, 13],
              3: [14, 15],
              4: [13],
              5: [6, 7, 9, 10, 11, 12, 14, 15],
              6: [5, 7, 10, 11, 14, 15, 16, 17],
              7: [5, 6, 9, 10, 11, 12, 14, 15],
              8: [16, 17, 18, 19],
              9: [5, 7, 10, 16, 17, 18, 19],
              10:[5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17],
              11:[5, 6, 7, 10, 13],
              12:[5, 7, 10, 16, 17],
              13:[1, 2, 4, 10, 11],
              14:[3, 5, 6, 7, 10],
              15:[3, 5, 6, 7, 10],
              16:[6, 8, 9, 10, 12, 18],
              17:[6, 8, 9, 10, 12, 19],
              18:[8, 9, 16],
              19:[8, 9, 17]}

atr_graph = {0: [],
              1: [2, 11],
              2: [1, 11],
              3: [11],
              4: [5, 6, 7, 11, 14, 15, 17],
              5: [4, 6, 7, 8, 12, 13],
              6: [4,5,7,8,9,10,12,13],
              7: [4,11,12,13,14,15],
              8: [5,6],
              9: [6, 12],
              10:[6, 13],
              11:[1,2,3,4,7,14,15,17],
              12:[5,6,7,9],
              13:[5,6,7,10],
              14:[4,7,11,16],
              15:[4,7,11,16],
              16:[14,15],
              17:[4,11],
              }

cihp2pascal_adj = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

cihp2pascal_nlp_adj = \
    np.array([[ 1.,  0.35333052,  0.32727194,  0.17418084,  0.18757584,
         0.40608522,  0.37503981,  0.35448462,  0.22598555,  0.23893579,
         0.33064262,  0.28923404,  0.27986573,  0.4211553 ,  0.36915778,
         0.41377746,  0.32485771,  0.37248222,  0.36865639,  0.41500332],
       [ 0.39615879,  0.46201529,  0.52321467,  0.30826114,  0.25669527,
         0.54747773,  0.3670523 ,  0.3901983 ,  0.27519473,  0.3433325 ,
         0.52728509,  0.32771333,  0.34819325,  0.63882953,  0.68042925,
         0.69368576,  0.63395791,  0.65344337,  0.59538781,  0.6071375 ],
       [ 0.16373166,  0.21663339,  0.3053872 ,  0.28377612,  0.1372435 ,
         0.4448808 ,  0.29479995,  0.31092595,  0.22703953,  0.33983576,
         0.75778818,  0.2619818 ,  0.37069392,  0.35184867,  0.49877512,
         0.49979437,  0.51853277,  0.52517541,  0.32517741,  0.32377309],
       [ 0.32687232,  0.38482461,  0.37693463,  0.41610834,  0.20415749,
         0.76749079,  0.35139853,  0.3787411 ,  0.28411737,  0.35155421,
         0.58792618,  0.31141718,  0.40585111,  0.51189218,  0.82042737,
         0.8342413 ,  0.70732188,  0.72752501,  0.60327325,  0.61431337],
       [ 0.34069369,  0.34817292,  0.37525998,  0.36497069,  0.17841617,
         0.69746208,  0.31731463,  0.34628951,  0.25167277,  0.32072379,
         0.56711286,  0.24894776,  0.37000453,  0.52600859,  0.82483993,
         0.84966274,  0.7033991 ,  0.73449378,  0.56649608,  0.58888791],
       [ 0.28477487,  0.35139564,  0.42742352,  0.41664321,  0.20004676,
         0.78566833,  0.42237487,  0.41048549,  0.37933812,  0.46542516,
         0.62444759,  0.3274493 ,  0.49466009,  0.49314658,  0.71244233,
         0.71497003,  0.8234787 ,  0.83566589,  0.62597135,  0.62626812],
       [ 0.3011378 ,  0.31775977,  0.42922647,  0.36896257,  0.17597556,
         0.72214655,  0.39162804,  0.38137872,  0.34980296,  0.43818419,
         0.60879174,  0.26762545,  0.46271161,  0.51150476,  0.72318109,
         0.73678399,  0.82620388,  0.84942166,  0.5943811 ,  0.60607602]])

pascal2atr_nlp_adj = \
    np.array([[ 1.,  0.35333052,  0.32727194,  0.18757584,  0.40608522,
         0.27986573,  0.23893579,  0.27600672,  0.30964391,  0.36865639,
         0.41500332,  0.4211553 ,  0.32485771,  0.37248222,  0.36915778,
         0.41377746,  0.32006291,  0.28923404],
       [ 0.39615879,  0.46201529,  0.52321467,  0.25669527,  0.54747773,
         0.34819325,  0.3433325 ,  0.26603942,  0.45162929,  0.59538781,
         0.6071375 ,  0.63882953,  0.63395791,  0.65344337,  0.68042925,
         0.69368576,  0.44354613,  0.32771333],
       [ 0.16373166,  0.21663339,  0.3053872 ,  0.1372435 ,  0.4448808 ,
         0.37069392,  0.33983576,  0.26563416,  0.35443504,  0.32517741,
         0.32377309,  0.35184867,  0.51853277,  0.52517541,  0.49877512,
         0.49979437,  0.21750868,  0.2619818 ],
       [ 0.32687232,  0.38482461,  0.37693463,  0.20415749,  0.76749079,
         0.40585111,  0.35155421,  0.28271333,  0.52684576,  0.60327325,
         0.61431337,  0.51189218,  0.70732188,  0.72752501,  0.82042737,
         0.8342413 ,  0.40137029,  0.31141718],
       [ 0.34069369,  0.34817292,  0.37525998,  0.17841617,  0.69746208,
         0.37000453,  0.32072379,  0.27268885,  0.47426719,  0.56649608,
         0.58888791,  0.52600859,  0.7033991 ,  0.73449378,  0.82483993,
         0.84966274,  0.37830796,  0.24894776],
       [ 0.28477487,  0.35139564,  0.42742352,  0.20004676,  0.78566833,
         0.49466009,  0.46542516,  0.32662614,  0.55780359,  0.62597135,
         0.62626812,  0.49314658,  0.8234787 ,  0.83566589,  0.71244233,
         0.71497003,  0.41223219,  0.3274493 ],
       [ 0.3011378 ,  0.31775977,  0.42922647,  0.17597556,  0.72214655,
         0.46271161,  0.43818419,  0.3192333 ,  0.50979216,  0.5943811 ,
         0.60607602,  0.51150476,  0.82620388,  0.84942166,  0.72318109,
         0.73678399,  0.39259827,  0.26762545]])

cihp2atr_nlp_adj = np.array([[ 1.,  0.35333052,  0.32727194,  0.18757584,  0.40608522,
         0.27986573,  0.23893579,  0.27600672,  0.30964391,  0.36865639,
         0.41500332,  0.4211553 ,  0.32485771,  0.37248222,  0.36915778,
         0.41377746,  0.32006291,  0.28923404],
       [ 0.35333052,  1.        ,  0.39206695,  0.42143438,  0.4736689 ,
         0.47139544,  0.51999208,  0.38354847,  0.45628529,  0.46514124,
         0.50083501,  0.4310595 ,  0.39371443,  0.4319752 ,  0.42938598,
         0.46384034,  0.44833757,  0.6153155 ],
       [ 0.32727194,  0.39206695,  1.        ,  0.32836702,  0.52603065,
         0.39543695,  0.3622627 ,  0.43575346,  0.33866223,  0.45202552,
         0.48421   ,  0.53669903,  0.47266611,  0.50925436,  0.42286557,
         0.45403656,  0.37221304,  0.40999322],
       [ 0.17418084,  0.46892601,  0.25774838,  0.31816231,  0.39330317,
         0.34218382,  0.48253904,  0.22084125,  0.41335728,  0.52437572,
         0.5191713 ,  0.33576117,  0.44230914,  0.44250678,  0.44330833,
         0.43887264,  0.50693611,  0.39278795],
       [ 0.18757584,  0.42143438,  0.32836702,  1.        ,  0.35030067,
         0.30110947,  0.41055555,  0.34338879,  0.34336307,  0.37704433,
         0.38810141,  0.34702081,  0.24171562,  0.25433078,  0.24696241,
         0.2570884 ,  0.4465962 ,  0.45263213],
       [ 0.40608522,  0.4736689 ,  0.52603065,  0.35030067,  1.        ,
         0.54372584,  0.58300258,  0.56674191,  0.555266  ,  0.66599594,
         0.68567555,  0.55716359,  0.62997328,  0.65638548,  0.61219615,
         0.63183318,  0.54464151,  0.44293752],
       [ 0.37503981,  0.50675565,  0.4761106 ,  0.37561813,  0.60419403,
         0.77912403,  0.64595517,  0.85939662,  0.46037144,  0.52348817,
         0.55875094,  0.37741886,  0.455671  ,  0.49434392,  0.38479954,
         0.41804074,  0.47285709,  0.57236283],
       [ 0.35448462,  0.50576632,  0.51030446,  0.35841033,  0.55106903,
         0.50257274,  0.52591451,  0.4283053 ,  0.39991808,  0.42327211,
         0.42853819,  0.42071825,  0.41240559,  0.42259136,  0.38125352,
         0.3868255 ,  0.47604934,  0.51811717],
       [ 0.22598555,  0.5053299 ,  0.36301185,  0.38002282,  0.49700941,
         0.45625243,  0.62876479,  0.4112051 ,  0.33944371,  0.48322639,
         0.50318714,  0.29207815,  0.38801966,  0.41119094,  0.29199072,
         0.31021029,  0.41594871,  0.54961962],
       [ 0.23893579,  0.51999208,  0.3622627 ,  0.41055555,  0.58300258,
         0.68874251,  1.        ,  0.56977937,  0.49918447,  0.48484363,
         0.51615925,  0.41222306,  0.49535971,  0.53134951,  0.3807616 ,
         0.41050298,  0.48675801,  0.51112664],
       [ 0.33064262,  0.306412  ,  0.60679935,  0.25592294,  0.58738706,
         0.40379627,  0.39679161,  0.33618385,  0.39235148,  0.45474013,
         0.4648476 ,  0.59306762,  0.58976007,  0.60778661,  0.55400397,
         0.56551297,  0.3698029 ,  0.33860535],
       [ 0.28923404,  0.6153155 ,  0.40999322,  0.45263213,  0.44293752,
         0.60359359,  0.51112664,  0.46578181,  0.45656936,  0.38142307,
         0.38525582,  0.33327223,  0.35360175,  0.36156453,  0.3384992 ,
         0.34261229,  0.49297863,  1.        ],
       [ 0.27986573,  0.47139544,  0.39543695,  0.30110947,  0.54372584,
         1.        ,  0.68874251,  0.67765588,  0.48690078,  0.44010641,
         0.44921156,  0.32321099,  0.48311542,  0.4982002 ,  0.39378102,
         0.40297733,  0.45309735,  0.60359359],
       [ 0.4211553 ,  0.4310595 ,  0.53669903,  0.34702081,  0.55716359,
         0.32321099,  0.41222306,  0.25721705,  0.36633509,  0.5397475 ,
         0.56429928,  1.        ,  0.55796926,  0.58842844,  0.57930828,
         0.60410597,  0.41615326,  0.33327223],
       [ 0.36915778,  0.42938598,  0.42286557,  0.24696241,  0.61219615,
         0.39378102,  0.3807616 ,  0.28089866,  0.48450394,  0.77400821,
         0.68813814,  0.57930828,  0.8856886 ,  0.81673412,  1.        ,
         0.92279623,  0.46969152,  0.3384992 ],
       [ 0.41377746,  0.46384034,  0.45403656,  0.2570884 ,  0.63183318,
         0.40297733,  0.41050298,  0.332879  ,  0.48799542,  0.69231828,
         0.77015091,  0.60410597,  0.79788484,  0.88232104,  0.92279623,
         1.        ,  0.45685017,  0.34261229],
       [ 0.32485771,  0.39371443,  0.47266611,  0.24171562,  0.62997328,
         0.48311542,  0.49535971,  0.32477932,  0.51486622,  0.79353556,
         0.69768738,  0.55796926,  1.        ,  0.92373745,  0.8856886 ,
         0.79788484,  0.47883134,  0.35360175],
       [ 0.37248222,  0.4319752 ,  0.50925436,  0.25433078,  0.65638548,
         0.4982002 ,  0.53134951,  0.38057074,  0.52403969,  0.72035243,
         0.78711147,  0.58842844,  0.92373745,  1.        ,  0.81673412,
         0.88232104,  0.47109935,  0.36156453],
       [ 0.36865639,  0.46514124,  0.45202552,  0.37704433,  0.66599594,
         0.44010641,  0.48484363,  0.39636574,  0.50175258,  1.        ,
         0.91320249,  0.5397475 ,  0.79353556,  0.72035243,  0.77400821,
         0.69231828,  0.59087008,  0.38142307],
       [ 0.41500332,  0.50083501,  0.48421,  0.38810141,  0.68567555,
         0.44921156,  0.51615925,  0.45156472,  0.50438158,  0.91320249,
         1.,  0.56429928,  0.69768738,  0.78711147,  0.68813814,
         0.77015091,  0.57698754,  0.38525582]])



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj)) # return a adjacency matrix of adj ( type is numpy)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])) #
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.todense()

def row_norm(inputs):
    """
    Row norm of the input.

    Args:
        inputs: (array): write your description
    """
    outputs = []
    for x in inputs:
        xsum = x.sum()
        x = x / xsum
        outputs.append(x)
    return outputs


def normalize_adj_torch(adj):
    """
    Normalize the adjacency matrix.

    Args:
        adj: (todo): write your description
    """
    # print(adj.size())
    if len(adj.size()) == 4:
        new_r = torch.zeros(adj.size()).type_as(adj)
        for i in range(adj.size(1)):
            adj_item = adj[0,i]
            rowsum = adj_item.sum(1)
            d_inv_sqrt = rowsum.pow_(-0.5)
            d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            r = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj_item), d_mat_inv_sqrt)
            new_r[0,i,...] = r
        return new_r
    rowsum = adj.sum(1)
    d_inv_sqrt = rowsum.pow_(-0.5)
    d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    r = torch.matmul(torch.matmul(d_mat_inv_sqrt,adj),d_mat_inv_sqrt)
    return r

# def row_norm(adj):




if __name__ == '__main__':
    a= row_norm(cihp2pascal_adj)
    print(a)
    print(cihp2pascal_adj)
    # print(a.shape)

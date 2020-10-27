import os.path as osp

data_dir = osp.join('data', 'weibo-1') # 数据集路径

if '1' in data_dir:
    # 话题
    topic = '#90后该不该攒钱来改变现状#'
    # weibo-1各部分nid范围
    labeled_nid = set(range(60))
    unlabeled_nid = set(range(60, 1623))
    test_nid = set(range(1623, 2123))
    labeled_nid_list = list(range(60))
    unlabeled_nid_list = list(range(60, 1623))
    test_nid_list = list(range(1623, 2123))
    # 句子长度
    LEN = {'seg':44, 'char':82}
elif '2' in data_dir:
    # 话题
    topic = '#停课不停学#'
    # weibo-2各部分nid范围
    labeled_nid = set(range(60))
    unlabeled_nid = set(range(60, 17195))
    test_nid = set(range(17195, 17695))
    labeled_nid_list = list(range(60))
    unlabeled_nid_list = list(range(60, 17195))
    test_nid_list = list(range(17195, 17695))
    # 句子长度
    LEN = {'seg':51, 'char':93}
elif '3' in data_dir:
    # 话题
    topic = '#建议把体育列入中高考必考科目#'
    # weibo-3各部分nid范围
    labeled_nid = set(range(60))
    unlabeled_nid = set(range(60, 5188))
    test_nid = set(range(5188, 5687))
    labeled_nid_list = list(range(60))
    unlabeled_nid_list = list(range(60, 5188))
    test_nid_list = list(range(5188, 5687))
    # 句子长度
    LEN = {'seg':38, 'char':65}
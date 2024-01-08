import os
import json
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np
import random

class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}
        self._feature_store_2 = {}

        self.val_unseen = ['2azQ1b91cZZ', '8194nk5LbLH', 'EU6Fwq7SyZv', 'oLBMNvg9in8', 'QUCTc6BB5sX', 'TbHJrupSAjP',
                           'X7HyMhZNoso', 'pLe4wQe7qrG', 'x8F5xyUWy9e', 'Z6MFQCViBuw', 'zsNo4HB9uLZ']

        self.test = ['2t7WUuJeko7', '5ZKStnWn8Zo', 'ARNzJeq3xxb', 'fzynW3qQPVF', 'jtcxE69GiFV', 'pa4otMbVnkk',
                     'q9vSo1VnCiC', 'rqfALeAoiTq', 'UwV83HsGsw3', 'wc2JMjhGNzB', 'WYY7iVyf5p8', 'YFuZgdQ5vWj',
                     'yqstnuAEVhm', 'YVUC4YcDtcY', 'gxdoqLR6rwA', 'gYvKGZ5eRqb', 'RPmz2sHmrrY', 'Vt2qJdWjCF2']

    def get_image_feature(self, scan, viewpoint, idx=None):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            # baseline
            # ft = self._feature_store[key]

            # print("idx: ", idx)

            ####### freq 数据增强时使用该代码 （轮流选择增强数据和原始数据）##########
            if scan in self.val_unseen or scan in self.test:
                ft = self._feature_store[key]
            elif idx % 2 == 0:
                ft = self._feature_store[key]
            else:
                ft = self._feature_store_2[key]

            # ##################### 两种特征 ##############################
            # ####### freq 数据增强时使用该代码 (随机选择增强数据和原始数据) ##########
            # if scan in self.val_unseen or scan in self.test:
            #     ft = self._feature_store[key]
            # elif random.randint(1,2) == 1:
            #     ft = self._feature_store[key]
            # else:
            #     ft = self._feature_store_2[key]

        else:
            # print(self.img_ft_file)
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_store[key] = ft

            # self.img_ft_file_2 = '/data/keji/Datasets/hamt_dataset/datasets/R2R/features/maxsim.hdf5'
            self.img_ft_file_2 = '/data/keji/Datasets/hamt_dataset/datasets/R2R/features/MASK3_samestylediffcontent2eachpoint_both-postive-negative.hdf5'
            with h5py.File(self.img_ft_file_2, 'r') as f:
                # ############ freq rgb_low_high_num=1 (rgb) ############
                # # (positive) RGB ft
                # ft_rgb_pos = f[key][...][:, :self.image_feat_size].astype(np.float32)
                # ft_pos = ft_rgb_pos
                # self._feature_store_2[key] = ft_pos

                # (negative) RGB ft
                ft_rgb_neg = f[key][...][:, self.image_feat_size + 1000:self.image_feat_size * 2 + 1000].astype(
                    np.float32)
                ft_neg = ft_rgb_neg
                self._feature_store_2[key] = ft_neg


        return ft

def get_obj_local_pos(raw_obj_pos):
    x1, y1, w, h = raw_obj_pos[:, 0], raw_obj_pos[:, 1], raw_obj_pos[:, 2], raw_obj_pos[:, 3]
    x2 = x1 + w
    y2 = y1 + h
    
    obj_local_pos = np.stack([x1/640, y1/480, x2/640, y2/480, w*h/(640*480)], 0).transpose()
    return obj_local_pos

def load_obj_database(obj_feat_file, image_feat_size):
    obj_feats = {}
    with h5py.File(obj_feat_file, 'r') as f:
        for key in f:
            obj_feats[key] = {
                'obj_ids': [str(x) for x in f[key].attrs['obj_ids']],
                'fts': f[key][...].astype(np.float32)[:, :image_feat_size],
                'bboxes': f[key].attrs['bboxes'],
                'viewindexs': f[key].attrs['viewindexs'],
            }
    return obj_feats
    
def load_instr_datasets(anno_dir, dataset, splits, tokenizer):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if tokenizer == 'bert':
                filepath = os.path.join(anno_dir, 'REVERIE_%s_enc.json' % split)
            elif tokenizer == 'xlm':
                filepath = os.path.join(anno_dir, 'REVERIE_%s_enc_xlmr.json' % split)
            else:
                raise NotImplementedError('unspported tokenizer %s' % tokenizer)

            with open(filepath) as f:
                new_data = json.load(f)
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)
        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer=None, max_instr_len=512):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            if 'objId' in item:
                new_item['instr_id'] = '%s_%s_%d' % (str(item['path_id']), str(item['objId']), j)
            else:
                new_item['path_id'] = item['id']
                new_item['instr_id'] = '%s_%d' % (item['id'], j)
                new_item['objId'] = None
            new_item['instruction'] = instr
            new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
            del new_item['instructions']
            del new_item['instr_encodings']

            # ''' BERT tokenizer '''
            # instr_tokens = ['[CLS]'] + tokenizer.tokenize(instr)[:max_instr_len-2] + ['[SEP]']
            # new_item['instr_encoding'] = tokenizer.convert_tokens_to_ids(instr_tokens)
                      
            data.append(new_item)
    return data


import os
import sys
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio # to load .mat files for depth points
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from model_util import datasetConfig

DC = datasetConfig() # dataset specific config
MAX_NUM_hand = 256
MEAN_COLOR_RGB = np.array([0.5,0.5,0.5])

class handposeDataset(Dataset):
    def __init__(self, split_set='train', num_points=20000,
        use_color=False, use_height=False, augment=False, scan_idx_list=None):

        assert(num_points<=50000)
        #self.data_path = os.path.join(ROOT_DIR, 'handpose/data/%s'%(split_set))
        self.data_path = '/handnet/data/%s'%(split_set)

        self.scan_names = sorted(list(set([os.path.basename(x)[0:6] \
            for x in os.listdir(self.data_path)])))
        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
       
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_hand,3) for GT hand point XYZ
            angle_class_label: (MAX_NUM_hand,) with int values in 0,...,NUM_ANGLE_BIN-1
            angle_residual_label: (MAX_NUM_hand,)
            size_classe_label: (MAX_NUM_hand,) with int values in 0,...,NUM_SIZE_CLUSTER
            hand_label_mask: (MAX_NUM_hand) as 0/1 with 1 indicating a unique hand
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_hands: unused
        """
        scan_name = self.scan_names[idx]
        point_cloud = np.load(os.path.join(self.data_path, scan_name)+'_pc.npz')['pc'] # Nx6
        hands = np.load(os.path.join(self.data_path, scan_name)+'_hand.npy') # K,8
        point_votes = np.load(os.path.join(self.data_path, scan_name)+'_votes.npz')['point_votes'] # Nx10

        if not self.use_color:
            point_cloud = point_cloud[:,0:3]
        else:
            point_cloud = point_cloud[:,0:6]
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)

        # ------------------------------- LABELS ------------------------------
        hand_centers = np.zeros((MAX_NUM_hand, 3))
        hand_sizes = np.zeros((MAX_NUM_hand, 3))
        angle_classes = np.zeros((MAX_NUM_hand,))
        angle_residuals = np.zeros((MAX_NUM_hand,))
        viewpoint_classes = np.zeros((MAX_NUM_hand,))
        widths = np.zeros((MAX_NUM_hand,))
        qualities = np.zeros((MAX_NUM_hand,))
        label_mask = np.zeros((MAX_NUM_hand))
        label_mask[0:hands.shape[0]] = 1

        for i in range(hands.shape[0]):
            hand = hands[i]
            hand_center = hand[0:3]
            viewpoint_class = hand[3]
            angle_class, angle_residual = DC.angle2class(hand[4])
            hand_quality = hand[5]
            hand_width = hand[6]
            semantic_class = hand[7]
            
            hand_centers[i,:] = hand_center
            viewpoint_classes[i] = viewpoint_class
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            qualities[i] = hand_quality
            widths[i] = hand_width

        target_hands_mask = label_mask 
        target_hands = np.zeros((MAX_NUM_hand, 6))
        for i in range(hands.shape[0]):
            hand = hands[i]
            target_hand = hand[0:6]
            target_hands[i,:] = target_hand

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_votes[choices,0]
        point_votes = point_votes[choices,1:]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)

        ret_dict['width_label'] = widths.astype(np.float32)
        ret_dict['quality_label'] = qualities.astype(np.float32)

        ret_dict['center_label'] = target_hands.astype(np.float32)[:,0:3]
        ret_dict['angle_class_label'] = angle_classes.astype(np.int64)
        ret_dict['angle_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['viewpoint_class_label'] = viewpoint_classes.astype(np.int64)
        ret_dict['hand_label_mask'] = target_hands_mask.astype(np.float32)
        
        return ret_dict
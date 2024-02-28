class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/ymz/newdisk2/workspace_tracking'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/ymz/newdisk2/workspace_tracking/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/ymz/newdisk2/workspace_tracking/pretrained_networks'
        self.lasot_dir = '/home/ymz/newdisk2/workspace_tracking/data/lasot'
        self.got10k_dir = '/home/ymz/newdisk2/workspace_tracking/data/got10k/train'
        self.got10k_val_dir = '/home/ymz/newdisk2/workspace_tracking/data/got10k/val'
        self.lasot_lmdb_dir = '/home/ymz/newdisk2/workspace_tracking/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/ymz/newdisk2/workspace_tracking/data/got10k_lmdb'
        self.trackingnet_dir = '/home/ymz/newdisk2/workspace_tracking/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/ymz/newdisk2/workspace_tracking/data/trackingnet_lmdb'
        self.coco_dir = '/home/ymz/newdisk2/workspace_tracking/data/coco'
        self.coco_lmdb_dir = '/home/ymz/newdisk2/workspace_tracking/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/ymz/newdisk2/workspace_tracking/data/vid'
        self.imagenet_lmdb_dir = '/home/ymz/newdisk2/workspace_tracking/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''

from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/ymz/newdisk2/workspace_tracking/data/got10k_lmdb'
    settings.got10k_path = '/home/ymz/newdisk2/workspace_tracking/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/ymz/newdisk2/workspace_tracking/data/itb'
    settings.lasot_extension_subset_path_path = '/home/ymz/newdisk2/workspace_tracking/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/ymz/newdisk2/workspace_tracking/data/lasot_lmdb'
    settings.lasot_path = '/home/ymz/newdisk2/workspace_tracking/data/lasot'
    settings.network_path = '/home/ymz/newdisk2/workspace_tracking/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/ymz/newdisk2/workspace_tracking/data/nfs'
    settings.otb_path = '/home/ymz/newdisk2/workspace_tracking/data/otb'
    settings.prj_dir = '/home/ymz/newdisk2/workspace_tracking'
    settings.result_plot_path = '/home/ymz/newdisk2/workspace_tracking/output/test/result_plots'
    settings.results_path = '/home/ymz/newdisk2/workspace_tracking/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/ymz/newdisk2/workspace_tracking/output'
    settings.segmentation_path = '/home/ymz/newdisk2/workspace_tracking/output/test/segmentation_results'
    settings.tc128_path = '/home/ymz/newdisk2/workspace_tracking/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/ymz/newdisk2/workspace_tracking/data/tnl2k/TNL2K_TEST'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/ymz/newdisk2/workspace_tracking/data/trackingnet'
    settings.uav_path = '/home/ymz/newdisk2/workspace_tracking/data/uav'
    settings.vot18_path = '/home/ymz/newdisk2/workspace_tracking/data/vot2018'
    settings.vot22_path = '/home/ymz/newdisk2/workspace_tracking/data/vot2022'
    settings.vot_path = '/home/ymz/newdisk2/workspace_tracking/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings


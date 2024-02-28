import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'lasot'
"""stark"""
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-S50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST101'))
"""TransT"""
# trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N2', result_only=True))
# trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N4', result_only=True))
"""pytracking"""
# trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
"""ostrack"""
# trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_distillation_123_128', dataset_name=dataset_name,
#                             run_ids=None, display_name='dist128'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_distillation_123_128_2', dataset_name=dataset_name,
#                             run_ids=None, display_name='dist128_2'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_distillation_123_128_h128', dataset_name=dataset_name,
#                             run_ids=None, display_name='dist128_head128'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ost_dist_128_h128_noce', dataset_name=dataset_name,
#                             run_ids=None, display_name='noce'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_distillation_123_128_h64', dataset_name=dataset_name,
#                             run_ids=None, display_name='128_64'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_distillation_123_128_h32', dataset_name=dataset_name,
#                             run_ids=None, display_name='128_32'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_distillation_123_64_h32', dataset_name=dataset_name,
#                             run_ids=None, display_name='64_32'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='ost128_h128', dataset_name=dataset_name,
#                             run_ids=None, display_name='new128_128'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128', dataset_name=dataset_name,
#                             run_ids=None, display_name='mae128'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128_3', dataset_name=dataset_name,
#                             run_ids=None, display_name='mae128_3'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mtr_128_h128_3', dataset_name=dataset_name,
#                             run_ids=None, display_name='mtr128_3'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128_9', dataset_name=dataset_name,
#                             run_ids=None, display_name='mae128_9'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128_3_Trblk', dataset_name=dataset_name,
#                             run_ids=None, display_name='TrBlk'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128_3_pe', dataset_name=dataset_name,
#                             run_ids=None, display_name='pe'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_pe_clip_1out', dataset_name=dataset_name,
#                             run_ids=None, display_name='clip'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_pe_clip_3out', dataset_name=dataset_name,
#                             run_ids=None, display_name='clip'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_64_h32_3_pe', dataset_name=dataset_name,
#                             run_ids=None, display_name='64_h32'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128_3_pe_KL', dataset_name=dataset_name,
#                             run_ids=None, display_name='KL'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='para_base_4_BN', dataset_name=dataset_name,
#                             run_ids=None, display_name='EV'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='vit_128_h128_noKD', dataset_name=dataset_name,
#                             run_ids=None, display_name='vit128'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128_noKD', dataset_name=dataset_name,
#                             run_ids=None, display_name='mae'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='base', dataset_name=dataset_name,
#                             run_ids=None, display_name='base'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='GAF', dataset_name=dataset_name,
#                             run_ids=None, display_name='GAF'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='GAF256', dataset_name=dataset_name,
#                             run_ids=None, display_name='GAF256'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='FGAF256', dataset_name=dataset_name,
#                             run_ids=None, display_name='FGAF256'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='AF256', dataset_name=dataset_name,
#                             run_ids=None, display_name='AF256'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='FAF256', dataset_name=dataset_name,
#                             run_ids=None, display_name='FAF256'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='FGAF256d4', dataset_name=dataset_name,
#                             run_ids=None, display_name='FGAF256d4'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='FGAF256d2', dataset_name=dataset_name,
#                             run_ids=None, display_name='FGAF256d2'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='mtr_FGAF256', dataset_name=dataset_name,
#                             run_ids=None, display_name='mtr_FGAF256'))
# # trackers.extend(trackerlist(name='efficienttrack', parameter_name='mtr_FGAF256_lr', dataset_name=dataset_name,
# #                             run_ids=None, display_name='mtr_FGAF256_lr'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='mtr_FGAF256_210', dataset_name=dataset_name,
#                             run_ids=None, display_name='mtr_FGAF256_210'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='mtr_FGAF256_500', dataset_name=dataset_name,
#                             run_ids=None, display_name='mtr_FGAF256_500'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='mtr_FGAF256_500_pefix', dataset_name=dataset_name,
#                             run_ids=None, display_name='mtr_FGAF256_500_pefix'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='FGAF256PE', dataset_name=dataset_name,
#                             run_ids=None, display_name='FGAF256PE'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='FGAF256PE2', dataset_name=dataset_name,
#                             run_ids=None, display_name='FGAF256PE2'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='FGAF256PE2_mtr200', dataset_name=dataset_name,
#                             run_ids=None, display_name='FGAF256PE2_mtr200'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='FGAF256PE2_mtr360', dataset_name=dataset_name,
#                             run_ids=None, display_name='FGAF256PE2_mtr360'))
# trackers.extend(trackerlist(name='vittrack', parameter_name='d12c256conv2', dataset_name=dataset_name,
#                             run_ids=None, display_name='d12c256conv2'))
# trackers.extend(trackerlist(name='vittrack', parameter_name='d12c256conv2_mtr320', dataset_name=dataset_name,
#                             run_ids=None, display_name='d12c256conv2_mtr320'))
# trackers.extend(trackerlist(name='vittrack', parameter_name='d12c256conv2_mtr320_lr1', dataset_name=dataset_name,
#                             run_ids=None, display_name='d12c256conv2_mtr320_lr1'))
# trackers.extend(trackerlist(name='vittrack', parameter_name='d12c256conv2_mtr_lr1', dataset_name=dataset_name,
#                             run_ids=None, display_name='d12c256conv2_mtr_lr1'))
# trackers.extend(trackerlist(name='vittrack', parameter_name='d12c256conv2_mtr', dataset_name=dataset_name,
#                             run_ids=None, display_name='d12c256conv2_mtr'))
# trackers.extend(trackerlist(name='vittrack', parameter_name='d12c256conv2_mtr_standard', dataset_name=dataset_name,
#                             run_ids=None, display_name='d12c256conv2_mtr_standard'))
# trackers.extend(trackerlist(name='vittrack', parameter_name='d12c256conv2_mtr_lr1_FPN', dataset_name=dataset_name,
#                             run_ids=None, display_name='d12c256conv2_mtr_lr1_FPN'))
# trackers.extend(trackerlist(name='vittrack', parameter_name='d12c256conv2_mtr_centerhead', dataset_name=dataset_name,
#                             run_ids=None, display_name='d12c256conv2_mtr_centerhead'))
# trackers.extend(trackerlist(name='vittrack', parameter_name='d12c256conv2_mtr_lr1_rect', dataset_name=dataset_name,
#                             run_ids=None, display_name='d12c256conv2_mtr_lr1_rect'))
# trackers.extend(trackerlist(name='vittrack', parameter_name='d12c256conv2_mtr_lr1_learnrect', dataset_name=dataset_name,
#                             run_ids=None, display_name='d12c256conv2_mtr_lr1_learnrect'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_learnrect', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_learnrect'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_colorfromtemplate', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_colorfromtemplate'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_templateembedding', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_templateembedding'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_templateembedding_b64', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_templateembedding_b64'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_colorfromtemplate_clip', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_colorfromtemplate_clip'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='FGAF', dataset_name=dataset_name,
#                             run_ids=None, display_name='FGAF'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='base', dataset_name=dataset_name,
#                             run_ids=None, display_name='base'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_jittered_search_embedding', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_jittered_search_embedding'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_td_osckpt', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_td_osckpt'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_talpha_osckpt', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_talpha_osckpt'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_osckpt', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_osckpt'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_talphamask_osckpt', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_talphamask_osckpt'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_extratempmask_osckpt', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_extratempmask_osckpt'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_tdmask_mae', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_tdmask_mae'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_etm_mae', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_etm_mae'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_talphamask_mae', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_talphamask_mae'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_td_osckpt1', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_td_osckpt1'))
# # trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_td_osckpt_nosig', dataset_name=dataset_name,
# #                             run_ids=None, display_name='vitb_td_osckpt_nosig'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_td_osckpt_tj', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_td_osckpt_tj'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_ce_td_osckpt', dataset_name=dataset_name,
#                             run_ids=None, display_name='vitb_ce_td_osckpt'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='viptb_ce_draw', dataset_name=dataset_name,
#                             run_ids=None, display_name='viptb_ce_draw'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='viptb_ce_draw_dim32', dataset_name=dataset_name,
#                             run_ids=None, display_name='viptb_ce_draw_dim32'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='viptb_ce_draw_dim32_freezeos', dataset_name=dataset_name,
#                             run_ids=None, display_name='viptb_ce_draw_dim32_freezeos'))
trackers.extend(trackerlist(name='ostrack', parameter_name='viptb_image', dataset_name=dataset_name,
                            run_ids=None, display_name='viptb_image'))
trackers.extend(trackerlist(name='ostrack', parameter_name='viptb_image_searchlearn', dataset_name=dataset_name,
                            run_ids=None, display_name='viptb_image_searchlearn'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384'))


dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))

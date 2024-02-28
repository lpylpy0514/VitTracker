from torch.nn.functional import l1_loss
from torch.nn import KLDivLoss
import torch
import torch.nn.functional as F

def aux_loss(cfg, teacher_res, student_res):
    if cfg.TRAIN.TEACHER == "MAE-L":
        loss = l1_loss(teacher_res['res_list'][23], student_res['res_list'][11]) + \
               l1_loss(teacher_res['res_list'][15], student_res['res_list'][7]) + \
               l1_loss(teacher_res['res_list'][7], student_res['res_list'][3])
    elif cfg.TRAIN.AUX_TYPE == "3 output" or cfg.TRAIN.AUX_TYPE == "Trblk":
        loss = l1_loss(teacher_res['res_list'][11], student_res['res_list'][2]) + \
            l1_loss(teacher_res['res_list'][7], student_res['res_list'][1]) + \
            l1_loss(teacher_res['res_list'][3], student_res['res_list'][0])
    elif cfg.TRAIN.AUX_TYPE == "1 output":
        loss = l1_loss(teacher_res['res_list'][11], student_res['res_list'][2])
    elif cfg.TRAIN.AUX_TYPE == "mean":
        feature_teacher = torch.mean(teacher_res['res_list'][11], dim=2)
        feature_student = torch.mean(student_res['res_list'][2], dim=2)
        loss = 768 * l1_loss(feature_teacher, feature_student)
    elif cfg.TRAIN.AUX_TYPE == "KLloss":
        loss_fun = KLDivLoss(reduction='mean')
        loss = loss_fun(F.log_softmax(teacher_res['res_list'][11], dim=2), F.softmax(student_res['res_list'][2])) + \
            loss_fun(F.log_softmax(teacher_res['res_list'][7], dim=2), F.softmax(student_res['res_list'][1])) + \
            loss_fun(F.log_softmax(teacher_res['res_list'][3], dim=2), F.softmax(student_res['res_list'][0]))

    return loss


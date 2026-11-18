import torch
from torch.nn import functional as F


def decoupled_kd_loss(student_logits, teacher_logits, labels, temperature,
                      target_weight, non_target_weight):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_logits = teacher_logits.detach()
    mask = F.one_hot(labels, num_classes=student_logits.shape[-1]).bool()
    student_prob = F.softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    student_binary = torch.stack((
        student_prob[mask], student_prob.masked_fill(mask, 0).sum(-1)), dim=-1)
    teacher_binary = torch.stack((
        teacher_prob[mask], teacher_prob.masked_fill(mask, 0).sum(-1)), dim=-1)
    target_loss = F.kl_div(
        student_binary.clamp_min(1e-8).log(), teacher_binary,
        reduction="batchmean") * temperature ** 2
    student_non_target = (student_logits / temperature).masked_fill(mask, -torch.inf)
    teacher_non_target = (teacher_logits / temperature).masked_fill(mask, -torch.inf)
    non_target_mask = ~mask
    student_log_prob = F.log_softmax(student_non_target, dim=-1)[non_target_mask]
    teacher_prob = F.softmax(teacher_non_target, dim=-1)[non_target_mask]
    non_target_loss = F.kl_div(
        student_log_prob, teacher_prob, reduction="sum"
    ) / student_logits.shape[0] * temperature ** 2
    return target_weight * target_loss + non_target_weight * non_target_loss

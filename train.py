import logging
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage import io

from utils import format_string, convert_from_color, count_sliding_window, grouper, sliding_window, CrossEntropy2d, dice_loss, \
    metrics, convert_to_color

# from loss.uncertainty import uncertainty_loss

logging.captureWarnings(True)
logger = logging.getLogger(__name__)
EPS = 1e-6


def test(dataset_cfg, training_cfg, model, test_ids, all=False, test_loader=None):
    # ===================== 加入 Ordos =====================
    if dataset_cfg.name == 'Potsdam' or dataset_cfg.name == 'Vaihingen' or dataset_cfg.name == 'Ordos':
        stride = dataset_cfg.stride_size
    batch_size = training_cfg.batch_size
    window_size = tuple(training_cfg.window_size)
    N_CLASSES = dataset_cfg.n_classes
    # Use the network on the test set
    if dataset_cfg.name == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(dataset_cfg.data_folder.format(id)), dtype='float32')
                       for id in
                       test_ids)
    elif dataset_cfg.name == 'Vaihingen':
        test_images = (1 / 255 * np.asarray(io.imread(dataset_cfg.data_folder.format(id)), dtype='float32') for id in
                       test_ids)
    # ===================== 新增 Ordos 图像读取 =====================
    elif dataset_cfg.name == 'Ordos':
        test_images = (1 / 255 * np.asarray(io.imread(dataset_cfg.data_folder.format(id)), dtype='float32') for id in
                       test_ids)

    if dataset_cfg.name == 'Potsdam':
        # dif_ids = [format_string(id) for id in test_ids]
        dif_ids = [id for id in test_ids]
        test_dsms = (np.asarray(io.imread(dataset_cfg.dsm_folder.format(id)), dtype='float32') for id in dif_ids)
    else:
        test_dsms = (np.asarray(io.imread(dataset_cfg.dsm_folder.format(id)), dtype='float32') for id in test_ids)

    invert_palette = {tuple(v): k for k, v in dataset_cfg.palette.items()}
    # 1. 改成列表推导式 [...]，并加上 [:, :, :3] 防止 PNG 的 Alpha 透明通道干扰颜色匹配
    test_labels = [convert_from_color(io.imread(dataset_cfg.label_folder.format(id))[:, :, :3], invert_palette) for id in test_ids]

    if dataset_cfg.name == 'Potsdam' or dataset_cfg.name == 'Vaihingen':
        eroded_labels = [convert_from_color(io.imread(dataset_cfg.eroded_folder.format(id))[:, :, :3], invert_palette) for id in test_ids]
    else:
        eroded_labels = test_labels  # 现在 test_labels 是列表了，直接赋值是绝对安全的

    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    # ===================== 加入 Ordos =====================
    if dataset_cfg.name == 'Potsdam' or dataset_cfg.name == 'Vaihingen' or dataset_cfg.name == 'Ordos':
        with torch.no_grad():
            for img, dsm, gt, gt_e in zip(test_images, test_dsms, test_labels, eroded_labels):
                pred = np.zeros(img.shape[:2] + (N_CLASSES,))

                total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
                for i, coords in enumerate(
                        grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))):
                    # Build the tensor
                    image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                    image_patches = np.asarray(image_patches)
                    image_patches = torch.from_numpy(image_patches).cuda()

                    min = np.min(dsm)
                    max = np.max(dsm)
                    dsm = (dsm - min) / (max - min)
                    dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                    dsm_patches = np.asarray(dsm_patches)
                    dsm_patches = torch.from_numpy(dsm_patches).cuda()

                    # Do the inference
                    outs, _, _, _, _ = model(image_patches, dsm_patches)
                    outs = outs.data.cpu().numpy()

                    # Fill in the results array
                    for out, (x, y, w, h) in zip(outs, coords):
                        out = out.transpose((1, 2, 0))
                        pred[x:x + w, y:y + h] += out
                    del (outs)

                pred = np.argmax(pred, axis=-1)
                all_preds.append(pred)
                all_gts.append(gt_e)
                # clear_output()

        results = metrics(np.concatenate([p.ravel() for p in all_preds]),
                          np.concatenate([p.ravel() for p in all_gts]).ravel(), dataset_cfg.labels,
                          dataset_cfg.n_classes)

        if all:
            return results, all_preds, all_gts
        else:
            return results


# def train(dataset_cfg, training_cfg, model, optimizer, scheduler, train_loader, weights, results_dir, test_loader=None):
#     weights = weights.cuda()
#     epochs = training_cfg.epochs
#     save_epoch = training_cfg.save_epoch
#     semantic_weight = getattr(training_cfg, "semantic_weight", 0.0)

#     history = {
#         'round': [],
#         'train_loss': [],
#         'Kappa': [],
#         'OA_total': [],
#         'MIoU_mean': [],
#         'F1_mean': []
#     }

#     for label in dataset_cfg.labels:
#         history[f'OA_{label}'] = []
#         history[f'MIoU_{label}'] = []
#         history[f'F1_{label}'] = []

#     MIoU_best = 0.0
#     epoch_best = -1
#     for epoch in range(1, epochs + 1):
#         logger.info('Train (epoch {}/{})'.format(epoch, epochs))
#         model.train()
#         batch_losses = []
#         total_iter = len(train_loader)
#         print_interval = max(1, total_iter // 10)
#         cached_num_pixels = None
#         log_num_pixels = None
#         for batch_idx, (opt, dsm, target) in enumerate(train_loader):
#             opt, dsm, target = opt.cuda(), dsm.cuda(), target.cuda()
#             optimizer.zero_grad()

#             output, L_cons, low_L_cons, semantic_prior, cca_loss = model(opt, dsm)
#             if torch.is_tensor(cca_loss):
#                 cca_loss = cca_loss.mean()
#             loss_ce = CrossEntropy2d(output, target, weight=weights)
#             loss_dice = dice_loss(output, target)
            
#             loss = loss_ce + (L_cons * training_cfg.alpha) - (low_L_cons * training_cfg.beta) + (loss_dice * training_cfg.gamma)
#             model_obj = getattr(model, "module", model)
#             cca_weight = getattr(model_obj, "cca_weight", 0.0)
#             if cca_loss is not None and cca_weight > 0:
#                 loss = loss + cca_weight * cca_loss
#             if semantic_prior is not None and semantic_weight > 0:
#                 log_probs = F.log_softmax(output, dim=1)
#                 num_pixels = output.shape[2] * output.shape[3]
#                 if cached_num_pixels != num_pixels:
#                     cached_num_pixels = num_pixels
#                     log_num_pixels = math.log(num_pixels)
#                 # log(mean(exp(log_probs))) for stable global class distribution.
#                 pred_log = torch.logsumexp(log_probs, dim=(2, 3)) - log_num_pixels
#                 target_prior = semantic_prior.clamp(min=EPS)
#                 # KL divergence between predicted class distribution and CLIP semantic prior.
#                 loss_sem = F.kl_div(pred_log, target_prior, reduction="batchmean", log_target=False)
#                 loss = loss + semantic_weight * loss_sem
#             loss.backward()
#             optimizer.step()

#             if scheduler is not None:
#                 scheduler.step()

#             batch_losses.append(loss.item())
#             if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == total_iter:
#                 print(f"Iter {batch_idx+1}/{total_iter} | Loss: {loss.item():.4f}")
#             del (opt, target, loss)

#         epoch_loss = np.mean(batch_losses)

#         if epoch % save_epoch == 0:
#             # We validate with the largest possible stride for faster computing
#             model.eval()
            
#             results_val = test(dataset_cfg, training_cfg, model, dataset_cfg.test_ids, all=False)
#             model.train()

#             MIoU = results_val['MIoU']['mean']

#             history['round'].append(epoch)
#             history['train_loss'].append(epoch_loss)
#             history['Kappa'].append(results_val['Kappa'])

#             history['OA_total'].append(results_val['OA']['total'])
#             for i in dataset_cfg.labels:
#                 history['OA_{}'.format(i)].append(results_val['OA'][i])

#             history['MIoU_mean'].append(results_val['MIoU']['mean'])
#             for i in dataset_cfg.labels:
#                 history['MIoU_{}'.format(i)].append(results_val['MIoU'][i])

#             history['F1_mean'].append(results_val['F1']['mean'])
#             for i in dataset_cfg.labels:
#                 history['F1_{}'.format(i)].append(results_val['F1'][i])

#             if MIoU > MIoU_best:
#                 # ===================== 新增 Ordos 最优模型保存 =====================
#                 if dataset_cfg.name == 'Vaihingen':
#                     torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_vaihingen'))
#                 elif dataset_cfg.name == 'Potsdam':
#                     torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_potsdam'))
#                 elif dataset_cfg.name == 'Ordos':
#                     torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_ordos'))

#                 MIoU_best = MIoU
#                 epoch_best = epoch

#             logger.info('    Training Loss: {}'.format(epoch_loss))
#             logger.info('    Kappa: {}'.format(results_val["Kappa"]))
#             logger.info('    OA: {}'.format(results_val["OA"]))
#             logger.info('    F1: {}'.format(results_val["F1"]))
#             logger.info('    MIoU: {}'.format(results_val["MIoU"]))
#             logger.info("")
#         else:
#             history['round'].append(epoch)
#             history['train_loss'].append(epoch_loss)
#             history['Kappa'].append(0.0)

#             history['OA_total'].append(0.0)
#             for i in dataset_cfg.labels:
#                 history['OA_{}'.format(i)].append(0.0)

#             history['MIoU_mean'].append(0.0)
#             for i in dataset_cfg.labels:
#                 history['MIoU_{}'.format(i)].append(0.0)

#             history['F1_mean'].append(0.0)
#             for i in dataset_cfg.labels:
#                 history['F1_{}'.format(i)].append(0.0)

#             logger.info('    Training Loss: {}'.format(epoch_loss))

#     logger.info('Best epoch {}, MIoU best: {}'.format(epoch_best, MIoU_best))

#     df = pd.DataFrame(history)
#     # ===================== 新增 Ordos 最终模型保存 =====================
#     if dataset_cfg.name == 'Vaihingen':
#         df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
#         torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_vaihingen'))
#     elif dataset_cfg.name == 'Potsdam':
#         df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
#         torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_potsdam'))
#     elif dataset_cfg.name == 'Ordos':
#         df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
#         torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_ordos'))

#     logger.info('End of training !')

def train(dataset_cfg, training_cfg, model, optimizer, scheduler, train_loader, weights, results_dir, test_loader=None):
    weights = weights.cuda()
    epochs = training_cfg.epochs
    save_epoch = training_cfg.save_epoch  # 原来的验证间隔
    save_interval = 10  # 每10个epoch保存一次模型  ✅核心在这里

    semantic_weight = getattr(training_cfg, "semantic_weight", 0.0)

    history = {
        'round': [],
        'train_loss': [],
        'Kappa': [],
        'OA_total': [],
        'MIoU_mean': [],
        'F1_mean': []
    }

    for label in dataset_cfg.labels:
        history[f'OA_{label}'] = []
        history[f'MIoU_{label}'] = []
        history[f'F1_{label}'] = []

    MIoU_best = 0.0
    epoch_best = -1
    for epoch in range(1, epochs + 1):
        logger.info('Train (epoch {}/{})'.format(epoch, epochs))
        model.train()
        batch_losses = []
        total_iter = len(train_loader)
        print_interval = max(1, total_iter // 10)
        cached_num_pixels = None
        log_num_pixels = None
        for batch_idx, (opt, dsm, target) in enumerate(train_loader):
            opt, dsm, target = opt.cuda(), dsm.cuda(), target.cuda()
            optimizer.zero_grad()

            output, L_cons, low_L_cons, semantic_prior, cca_loss = model(opt, dsm)
            if torch.is_tensor(cca_loss):
                cca_loss = cca_loss.mean()
            loss_ce = CrossEntropy2d(output, target, weight=weights)
            loss_dice = dice_loss(output, target)
            
            loss = loss_ce + (L_cons * training_cfg.alpha) - (low_L_cons * training_cfg.beta) + (loss_dice * training_cfg.gamma)
            model_obj = getattr(model, "module", model)
            cca_weight = getattr(model_obj, "cca_weight", 0.0)
            if cca_loss is not None and cca_weight > 0:
                loss = loss + cca_weight * cca_loss
            if semantic_prior is not None and semantic_weight > 0:
                log_probs = F.log_softmax(output, dim=1)
                num_pixels = output.shape[2] * output.shape[3]
                if cached_num_pixels != num_pixels:
                    cached_num_pixels = num_pixels
                    log_num_pixels = math.log(num_pixels)
                pred_log = torch.logsumexp(log_probs, dim=(2, 3)) - log_num_pixels
                target_prior = semantic_prior.clamp(min=EPS)
                loss_sem = F.kl_div(pred_log, target_prior, reduction="batchmean", log_target=False)
                loss = loss + semantic_weight * loss_sem
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            batch_losses.append(loss.item())
            if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == total_iter:
                print(f"Iter {batch_idx+1}/{total_iter} | Loss: {loss.item():.4f}")
            del (opt, target, loss)

        epoch_loss = np.mean(batch_losses)

        if epoch % save_epoch == 0:
            model.eval()
            results_val = test(dataset_cfg, training_cfg, model, dataset_cfg.test_ids, all=False)
            model.train()

            MIoU = results_val['MIoU']['mean']

            history['round'].append(epoch)
            history['train_loss'].append(epoch_loss)
            history['Kappa'].append(results_val['Kappa'])

            history['OA_total'].append(results_val['OA']['total'])
            for i in dataset_cfg.labels:
                history['OA_{}'.format(i)].append(results_val['OA'][i])

            history['MIoU_mean'].append(results_val['MIoU']['mean'])
            for i in dataset_cfg.labels:
                history['MIoU_{}'.format(i)].append(results_val['MIoU'][i])

            history['F1_mean'].append(results_val['F1']['mean'])
            for i in dataset_cfg.labels:
                history['F1_{}'.format(i)].append(results_val['F1'][i])

            # ===================== 保存最优模型 best_model ✅
            if MIoU > MIoU_best:
                if dataset_cfg.name == 'Vaihingen':
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_vaihingen.pth'))
                elif dataset_cfg.name == 'Potsdam':
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_potsdam.pth'))
                elif dataset_cfg.name == 'Ordos':
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_ordos.pth'))

                MIoU_best = MIoU
                epoch_best = epoch

            logger.info('    Training Loss: {}'.format(epoch_loss))
            logger.info('    Kappa: {}'.format(results_val["Kappa"]))
            logger.info('    OA: {}'.format(results_val["OA"]))
            logger.info('    F1: {}'.format(results_val["F1"]))
            logger.info('    MIoU: {}'.format(results_val["MIoU"]))
            logger.info("")
        else:
            history['round'].append(epoch)
            history['train_loss'].append(epoch_loss)
            history['Kappa'].append(0.0)

            history['OA_total'].append(0.0)
            for i in dataset_cfg.labels:
                history['OA_{}'.format(i)].append(0.0)

            history['MIoU_mean'].append(0.0)
            for i in dataset_cfg.labels:
                history['MIoU_{}'.format(i)].append(0.0)

            history['F1_mean'].append(0.0)
            for i in dataset_cfg.labels:
                history['F1_{}'.format(i)].append(0.0)

            logger.info('    Training Loss: {}'.format(epoch_loss))

        # ===================== 每10个epoch 保存一次模型 ✅ 核心新增
        if epoch % save_interval == 0:
            save_name = f"epoch_{epoch}.pth"
            save_path = os.path.join(results_dir, save_name)
            torch.save(model.state_dict(), save_path)
            logger.info(f"已保存第 {epoch} 轮模型：{save_name}")

    # ===================== 训练结束 保存 final 模型 ✅
    logger.info('Best epoch {}, MIoU best: {}'.format(epoch_best, MIoU_best))

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)

    if dataset_cfg.name == 'Vaihingen':
        torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_vaihingen.pth'))
    elif dataset_cfg.name == 'Potsdam':
        torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_potsdam.pth'))
    elif dataset_cfg.name == 'Ordos':
        torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_ordos.pth'))

    logger.info('End of training !')

def visualize_testloader(model, test_loader, palette, save_root):
    # os.makedirs(save_root, exist_ok=True)
    model.cuda()
    model.eval()
    tile_idx = 0
    with torch.no_grad():
        for img, dsm, _ in test_loader:
            img, dsm = img.cuda(), dsm.cuda()
            pred, _, _, _, _ = model(img, dsm)
            pred = pred.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            for i in range(pred.shape[0]):
                color_pred = convert_to_color(pred[i], palette)
                io.imsave(os.path.join(save_root, f"tile_{tile_idx}.png"),
                          color_pred, check_contrast=False)
                tile_idx += 1
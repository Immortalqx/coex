import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from ruamel.yaml import YAML
from stereo import Stereo

from dataloaders.stereo import AirsimListLoader as ASL
from dataloaders.stereo import AirsimLoader as AL

torch.backends.cudnn.benchmark = True

torch.set_grad_enabled(False)

config = 'cfg_coex.yaml'
version = 0  # CoEx
half_precision = True


def load_configs(path):
    cfg = YAML().load(open(path, 'r'))
    backbone_cfg = YAML().load(
        open(cfg['model']['stereo']['backbone']['cfg_path'], 'r'))
    cfg['model']['stereo']['backbone'].update(backbone_cfg)
    return cfg


if __name__ == '__main__':
    cfg = load_configs(
        './configs/stereo/{}'.format(config))
    # ckpt = '{}/{}/version_{}/checkpoints/last.ckpt'.format(
    #     'logs/stereo', cfg['model']['name'], version)
    # cfg['stereo_ckpt'] = ckpt
    # pose_ssstereo = Stereo.load_from_checkpoint(cfg['stereo_ckpt'], strict=False, cfg=cfg).cuda()
    pose_ssstereo = Stereo.load_from_checkpoint(
        "/home/immortalqx/Projects/coex/airsim/airsim.ckpt", strict=False,
        cfg=cfg).cuda()

    # 加载airsim数据集
    airsimpath = cfg['training']['paths']['airsim']
    (left_train, right_train, disp_train_L, disp_train_R,
     left_test, right_test, disp_test_L, disp_test_R) = ASL.listloader(airsimpath)

    airsimtrain = AL.ImageLoader(
        left_train, right_train, disp_train_L, disp_train_R, False, th=0, tw=0, load_raw=True)

    airsimtrain = DataLoader(
        airsimtrain, batch_size=cfg['training']['batch_size'],
        num_workers=8, shuffle=False, drop_last=False)

    fps_list = np.array([])
    pose_ssstereo.eval()
    for i, batch in enumerate(airsimtrain):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        imgL, imgR = batch['imgL'].cuda(), batch['imgR'].cuda()
        imgLRaw = batch['imgLRaw']
        imgLRaw = imgLRaw.cuda()

        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        print('Data Preparation: {:.3f}'.format(runtime))

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=half_precision):
                img = torch.cat([imgL, imgR], 0)
                disp = pose_ssstereo(img, training=False)
        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        # print('Stereo runtime: {:.3f}'.format(runtime))
        
        print(disp)
        print(disp.size)

        fps = 1000 / runtime
        fps_list = np.append(fps_list, fps)
        if len(fps_list) > 5:
            fps_list = fps_list[-5:]
        avg_fps = np.mean(fps_list)
        print('Stereo runtime: {:.3f}'.format(1000 / avg_fps))

        disp_np = disp[0].data.cpu().numpy().astype(np.uint8)
        disp_np = cv2.applyColorMap(disp_np * 1, cv2.COLORMAP_JET)

        image_np = (imgLRaw[0].permute(1, 2, 0).data.cpu().numpy()).astype(np.uint8)

        out_img = np.concatenate((image_np, disp_np), 0)
        cv2.putText(
            out_img,
            "%.1f fps" % (avg_fps),
            (10, image_np.shape[0] + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('img', out_img)
        cv2.waitKey(1)

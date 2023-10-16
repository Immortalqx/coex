from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger

# from models import *
from dataloaders.stereo import AirsimListLoader as ASL
from dataloaders.stereo import AirsimLoader as AL
from stereo import Stereo, load_configs, copy_dir

configs = [
    'cfg_coex.yaml',
    'cfg_psm.yaml'
]
config_num = 0


def train_airsim():
    pl.seed_everything(36)
    cfg = load_configs('./configs/stereo/{}'.format(configs[config_num]))
    logging_pth = cfg['training']['paths']['logging']
    th, tw = cfg['training']['th'], cfg['training']['tw']

    ''' Airsim Training Part '''
    if cfg['training']['train_on']['airsim']:
        # 加载airsim数据集
        airsimpath = cfg['training']['paths']['airsim']
        (left_train, right_train, disp_train_L, disp_train_R,
         left_test, right_test, disp_test_L, disp_test_R) = ASL.listloader(airsimpath)

        airsimtrain = AL.ImageLoader(
            left_train, right_train, disp_train_L, disp_train_R, True, th=th, tw=tw)
        airsimtrain = DataLoader(
            airsimtrain, batch_size=cfg['training']['batch_size'],
            num_workers=16, shuffle=True, drop_last=False)

        airsimtest = AL.ImageLoader(
            left_test, right_test, disp_test_L, disp_test_R, False)
        airsimtest = DataLoader(
            airsimtest, batch_size=1, num_workers=16,
            shuffle=False, drop_last=False)

        # Model
        # cfg = load_configs(
        #     './configs/stereo/{}'.format('cfg_coex.yaml'))
        # ckpt = '{}/{}/version_{}/checkpoints/last.ckpt'.format(
        #     'logs/stereo', cfg['model']['name'], 0)
        # cfg['stereo_ckpt'] = ckpt
        # stereo = Stereo.load_from_checkpoint(cfg['stereo_ckpt'], strict=False, cfg=cfg).cuda()
        # stereo.dataname = "sceneflow"  # 乱选一个，尝试下
        log_name = 'sceneflow'
        if cfg['training']['load_version'] is not None:
            load_version = cfg['training']['load_version']
            ckpt = '{}/{}/version_{}/checkpoints/sceneflow-epoch={}.ckpt'.format(
                logging_pth, cfg['model']['name'], load_version,
                cfg['training']['sceneflow_max_epochs'] - 1)
            stereo = Stereo.load_from_checkpoint(ckpt, cfg=cfg, dataname=log_name)
        else:
            stereo = Stereo(cfg, 'sceneflow')

        version = copy_dir(logging_pth, cfg['model']['name'], cfg['training']['save_version'])

        logger = TestTubeLogger(
            logging_pth,
            cfg['model']['name'],
            version=version)
        gpu_stats = pl.callbacks.GPUStatsMonitor()
        lr_monitor = pl.callbacks.LearningRateMonitor()
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=log_name + '-{epoch}',
            save_last=True,
            save_top_k=-1,
            monitor=log_name + '_train_loss_epoch')
        trainer = pl.Trainer(
            accelerator='dp',
            logger=logger,
            callbacks=[gpu_stats, lr_monitor, checkpoint_callback],
            precision=cfg['precision'],
            gpus=cfg['device'],
            max_epochs=cfg['training']['sceneflow_max_epochs'],
            gradient_clip_val=0.1,
            weights_summary='full',
        )

        trainer.fit(stereo, airsimtrain, airsimtest)
        trainer.test(stereo, airsimtest)


if __name__ == "__main__":
    train_airsim()

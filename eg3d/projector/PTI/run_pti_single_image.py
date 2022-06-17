from random import choice
from string import ascii_uppercase
from torchvision.transforms import transforms
import os
from configs import global_config, paths_config
import glob

from training.coaches.single_image_coach import SingleImageCoach


def run_PTI(run_name='', use_wandb=False, use_multi_id_training=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name


    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    coach = SingleImageCoach(trans)

    latent_space = 'w_plus'
    for image_path in glob.glob('../../projector_test_data/*.png'):
        name = os.path.basename(image_path)[:-4]
        w_path = f'../../projector_out/{name}_{latent_space}/{name}_{latent_space}.npy'
        c_path = f'../../projector_test_data/{name}.npy'
        if len(glob.glob(f'./checkpoints/*_{name}_{latent_space}.pth'))>0:
            continue

        if not os.path.exists(w_path):
            continue
        coach.train(image_path = image_path, w_path=w_path,c_path = c_path)

    latent_space = 'w'
    for image_path in glob.glob('../../projector_test_data/*.png'):
        name = os.path.basename(image_path)[:-4]
        w_path = f'../../projector_out/{name}_{latent_space}/{name}_{latent_space}.npy'
        c_path = f'../../projector_test_data/{name}.npy'
        if len(glob.glob(f'./checkpoints/*_{name}_{latent_space}.pth')) > 0:
            continue

        if not os.path.exists(w_path):
            continue
        coach.train(image_path=image_path, w_path=w_path, c_path=c_path)

    return global_config.run_name


if __name__ == '__main__':
    run_PTI(run_name='', use_wandb=False, use_multi_id_training=False)

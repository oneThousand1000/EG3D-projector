import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
import numpy as np
from PIL import Image

class SingleImageCoach(BaseCoach):

    def __init__(self,trans):
        super().__init__(data_loader=None, use_wandb=False)
        self.source_transform = trans

    def train(self, image_path, w_path,c_path):

        use_ball_holder = True

        name = os.path.basename(w_path)[:-4]
        print("image_path: ", image_path, 'c_path', c_path)
        c = np.load(c_path)

        c = np.reshape(c, (1, 25))

        c = torch.FloatTensor(c).cuda()

        from_im = Image.open(image_path).convert('RGB')

        if self.source_transform:
            image = self.source_transform(from_im)

        self.restart_training()




        print('load pre-computed w from ', w_path)
        if not os.path.isfile(w_path):
            print(w_path, 'is not exist!')
            return None

        w_pivot = torch.from_numpy(np.load(w_path)).to(global_config.device)


        # w_pivot = w_pivot.detach().clone().to(global_config.device)
        w_pivot = w_pivot.to(global_config.device)

        log_images_counter = 0
        real_images_batch = image.to(global_config.device)

        for i in tqdm(range(hyperparameters.max_pti_steps)):

            generated_images = self.forward(w_pivot, c)
            loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, name,
                                                           self.G, use_ball_holder, w_pivot)

            self.optimizer.zero_grad()

            if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                break

            loss.backward()
            self.optimizer.step()

            use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0


            global_config.training_step += 1
            log_images_counter += 1

        self.image_counter += 1

        save_dict = {
            'G_ema': self.G.state_dict()
        }
        checkpoint_path = f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{name}.pth'
        print('final model ckpt save to ', checkpoint_path)
        torch.save(save_dict, checkpoint_path)

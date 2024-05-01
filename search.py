import os
import yaml
import numpy as np


for pos_to_neg_loss_factor in np.arange(0.8, 2, 0.2):
    for learing_raate in [0.01, 0.007, 0.0001, 0.00001]:
        for maxthd in np.arange(0.25, 1, 0.2):
            for minthd in np.arange(0.1, 0.5, 0.1):
                output = f"{learing_raate}_{maxthd}_{minthd}_{pos_to_neg_loss_factor}"
                with open('configs/new_creative_pet.yaml', 'r+') as file:
                    data = yaml.load(file, Loader=yaml.FullLoader)
                    data['min_cosine_thr'] = minthd
                    data['max_cosine_thr'] = maxthd
                    data['pos_to_neg_loss_factor'] = pos_to_neg_loss_factor
                    data['learing_raate'] = learing_raate
                    yaml.dump(data, file)
                os.system(f"python -m scripts.train --config configs/new_creative_pet.yaml --output_dir=./{output}")
import os
import yaml
import numpy as np

for maxthd in np.arange(0.25, 0.31, 0.01):
    for minthd in np.arange(0.15, 0.21, 0.01):
        for pos_to_neg_loss_factor in np.arange(0.8, 2, 0.2):
            for learing_raate in [0.01, 0.007, 0.0001, 0.00001]:
                output = f"{learing_raate}_{maxthd}_{minthd}_{pos_to_neg_loss_factor}"
                completion_marker = os.path.join(output, 'training_done.txt')

                if not os.path.exists(completion_marker):
                    with open('configs/new_creative_pet.yaml', 'r+') as file:
                        data = yaml.load(file, Loader=yaml.FullLoader)
                        data['min_cosine_thr'] = float(minthd)
                        data['max_cosine_thr'] = float(maxthd)
                        data['pos_to_neg_loss_factor'] = float(pos_to_neg_loss_factor)
                        data['learning_rate'] = float(learing_raate)
                        file.seek(0)
                        file.truncate()
                        yaml.dump(data, file)
                    
                    os.system(f"python -m scripts.train --config configs/new_creative_pet.yaml --output_dir=./{output}")
                    
                    with open(completion_marker, 'w') as f:
                        f.write("Training completed successfully.")
                else:
                    print(f"Skipping {output} as training is already completed.")

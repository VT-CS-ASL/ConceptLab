import os
import yaml
import numpy as np

for maxthd in np.arange(0.25, 0.29, 0.03):
    for minthd in np.arange(0.15, 0.18, 0.01):
        for pos_to_neg_loss_factor in np.arange(0.8, 1.5, 0.2):
            for learing_raate in [0.01, 0.001, 0.0001]:
                for num_knn_class in [10, 7, 5]:

                    folder = ""
                    with open('configs/new_creative_pet.yaml', 'r+') as file:
                        data = yaml.load(file, Loader=yaml.FullLoader)
                        data['min_cosine_thr'] = float(minthd)
                        data['max_cosine_thr'] = float(maxthd)
                        data['pos_to_neg_loss_factor'] = float(pos_to_neg_loss_factor)
                        data['learning_rate'] = float(learing_raate)
                        data['num_knn_class'] = num_knn_class
                        folder = data['initializer_token']
                        file.seek(0)
                        file.truncate()
                        yaml.dump(data, file)

                    if folder and not os.path.exists(folder):
                        os.makedirs(folder)

                    output = f"{learing_raate}_{maxthd}_{minthd}_{pos_to_neg_loss_factor}_{num_knn_class}"
                    if folder:
                        output = folder + "/" + output
                    completion_marker = os.path.join(output, 'training_done.txt')

                    if not os.path.exists(completion_marker):
                        os.system(f"python -m scripts.train --config configs/new_creative_pet.yaml --output_dir=./{output}")
                        with open(completion_marker, 'w') as f:
                            f.write("Training completed successfully.")
                    else:
                        print(f"Skipping {output} as training is already completed.")

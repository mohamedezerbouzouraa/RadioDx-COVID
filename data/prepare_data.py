import os
import random
import shutil

CLASS_NAMES = ['normal', 'viral', 'covid']
ROOT_DIR = 'COVID-19 Radiography Database'
SOURCE_DIRS = ['NORMAL', 'Viral Pneumonia', 'COVID-19']
IMAGES_PER_CLASS_FOR_TEST = 30


def prepare_dataset(root_dir=ROOT_DIR, source_dirs=SOURCE_DIRS,
                     class_names=CLASS_NAMES,
                     n_test_images=IMAGES_PER_CLASS_FOR_TEST):
    if not os.path.isdir(os.path.join(root_dir, source_dirs[1])):
        print('Dataset already prepared, skipping.')
        return

    os.mkdir(os.path.join(root_dir, 'test'))

    for i, d in enumerate(source_dirs):
        os.rename(os.path.join(root_dir, d), os.path.join(root_dir, class_names[i]))

    for c in class_names:
        os.mkdir(os.path.join(root_dir, 'test', c))

    for c in class_names:
        images = [x for x in os.listdir(os.path.join(root_dir, c)) if x.lower().endswith('png')]
        selected_images = random.sample(images, n_test_images)
        for image in selected_images:
            source_path = os.path.join(root_dir, c, image)
            target_path = os.path.join(root_dir, 'test', c, image)
            shutil.move(source_path, target_path)

    print('Dataset prepared successfully.')


if __name__ == '__main__':
    prepare_dataset()

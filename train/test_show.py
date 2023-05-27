from dataset.carla_dataset import CarlaMVDetDataset
import logging
import cv2,time,torch
import numpy as np
logging.basicConfig(level=logging.INFO)

ds = CarlaMVDetDataset(
    root = 'train/test/test_data',
    towns = [1],
    weathers= [0],
    rgb_transform=None,
    )

# image = ds[30][1][0].numpy().astype(np.int8).reshape(100,100)
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
for i in range(len(ds)):
    image = np.array(ds[i][0]['rgb'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    cv2.imshow('image', image)
    if not torch.equal(torch.zeros(7),ds[i][1][-3]):
        logging.info('stop:%d' %i)
        logging.info(ds[i][1][-3])
        cv2.waitKey(0)
    else:
        cv2.waitKey(33)

cv2.waitKey(0)
cv2.destroyAllWindows()

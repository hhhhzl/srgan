from PIL import Image
import os
import random
import cv2


class PreProcessData():
    def __init__(self, data_list_url, dataset_root):
        self.data_list_url = data_list_url
        self.hr_train_name = 'train_HR'
        self.hr_valid_name = 'valid_HR'
        self.lr_train_name = 'train_LR'
        self.lr_valid_name = 'valid_LR'
        self.train_p = 0.7
        self.valid_p = 0.3
        self.test_p = 1 - self.valid_p - self.train_p
        # self.dataset_root = os.path.join(os.path.dirname(__file__), 'superReso')
        self.dataset_root = dataset_root
        self.valid_data_set = {
            'A020_KY': 1,
            'A020_WS': 1,
            'D144_KY': 1,
            'D144_WS': 1,
            'D149_WS-1': 1,
            'D149_WS-2': 1,
            'R_R002': 1,
            'R_R003': 1,
            'R_R004': 1,
        }

    @staticmethod
    # 转换图像为灰度
    def _to_gray_scale(file_path):
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            image = Image.open(file_path)
            grayscale_image = image.convert("L")
            return grayscale_image

    @staticmethod
    def _hr_to_lr(hr_file):
        hr = cv2.imread(hr_file, -1)
        hr_height, hr_width = hr.shape[:2]
        # 缩小图像
        shrink = cv2.resize(hr, (0, 0), fx=0.5, fy=0.5)

        # 放大图片
        lr = cv2.resize(shrink, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_AREA)
        cv2.imshow("hr", hr)
        cv2.imshow("lr", lr)
        return lr

    def _create_pic_folders(self):
        if not os.path.exists(self.dataset_root):
            os.mkdir(self.dataset_root)

        pic_list = [self.hr_train_name, self.hr_valid_name,
                    self.lr_train_name, self.lr_valid_name]

        for i in pic_list:
            if not os.path.exists(os.path.join(self.dataset_root, i)):
                os.mkdir(os.path.join(self.dataset_root, i))

    def _create_dataset(self):
        pic_count = 0
        pic_dic = {}
        for root, dirs, files in os.walk(self.data_list_url):
            for name in files:
                if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')) and root.split(os.sep)[-1] in self.valid_data_set:
                    pic_dic[pic_count] = os.path.join(root, name)
                    pic_count += 1

        pic_list = random.sample(range(pic_count), pic_count)
        train_list = pic_list[:int(pic_count * self.train_p)]
        valid_list = pic_list[int(pic_count * self.train_p):]

        for each_id in train_list:
            image_name = pic_dic[each_id].split(os.sep)[-1]
            gray_pic = self._to_gray_scale(pic_dic[each_id])
            gray_pic.save(os.path.join(self.dataset_root, f'{self.hr_train_name}/{image_name}'))
            lr = self._hr_to_lr(pic_dic[each_id])
            cv2.imwrite(os.path.join(self.dataset_root, f'{self.lr_train_name}/{image_name}'), lr)

        for each_id in valid_list:
            image_name = pic_dic[each_id].split(os.sep)[-1]
            gray_pic = self._to_gray_scale(pic_dic[each_id])
            gray_pic.save(os.path.join(self.dataset_root, f'{self.hr_valid_name}/{image_name}'))
            lr = self._hr_to_lr(pic_dic[each_id])
            cv2.imwrite(os.path.join(self.dataset_root, f'{self.lr_valid_name}/{image_name}'), lr)


if __name__ == "__main__":
    p = PreProcessData(
        # local set
        # data_list_url='/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/Research/SuperRevolution/existModels/SRGAN/processing_list',

        data_list_url='data/dataset/processing_list',
        dataset_root='data/dataset/superReso'
    )
    p._create_pic_folders()
    p._create_dataset()

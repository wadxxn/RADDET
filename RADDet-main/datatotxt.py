

import os

def write_dataset2txt(dataset_path, save_path):
    '''
    :param save_path: txt文件保存的目标路径
    :return:
    '''
    # 分类文件夹名称
    classes_name = os.listdir(dataset_path)  # 列表形式存储
    print(f'classes_name: {classes_name}')
 
    # 执行写入文件操作，如果文件已存在，则不执行写入操作，需手动删除文件后再执行
    if os.path.exists(save_path):
        print(f'{save_path} already exists! Please delete it first.')
    else:
        for classes in classes_name:
            cls_path = f'{dataset_path}/{classes}'
            for i in os.listdir(cls_path):
                img_path = f'{cls_path}/{i}'
                with open(os.path.join(save_path), "a+", encoding="utf-8", errors="ignore") as f:
                    f.write(img_path + '\n')
        print('Writing dataset to file is finish!')
dataset_path = r'D:/2022-02-Deep Radar Detector/RADDet/train/RAD'
dataset_txt_path = r'dataset.txt'
write_dataset2txt(dataset_path, dataset_txt_path)
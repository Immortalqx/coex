import os
import os.path


def airsimfolderloader(folderpath):
    left_list = []
    right_list = []
    disp_L_list = []
    disp_R_list = []
    file_list = open(folderpath + "/airsim_rec.txt", "r").readlines()
    for i in range(len(file_list)):
        if i == 0:  # 跳过第一行
            continue
        # 根据"\t"划分，最后一项是所有录制的文件，再根据";"划分，得到（左目、左目深度、右目、右目深度+"\n）
        all_name = file_list[i].split("\t")[-1].split(";")
        left_list.append(folderpath + "/images/" + all_name[0])
        right_list.append(folderpath + "/images/" + all_name[2])
        disp_L_list.append(folderpath + "/images/" + all_name[1])
        disp_R_list.append(folderpath + "/images/" + all_name[3].split("\n")[0])
    return left_list, right_list, disp_L_list, disp_R_list


def listloader(filepath):
    # 加载训练数据
    train_path = os.path.join(filepath, "train")
    train_dir = os.listdir(train_path)

    left_train = []
    right_train = []
    disp_train_L = []
    disp_train_R = []

    for single_dir in train_dir:
        left_list, right_list, disp_L_list, disp_R_list = airsimfolderloader(os.path.join(train_path, single_dir))
        left_train.extend(left_list)
        right_train.extend(right_list)
        disp_train_L.extend(disp_L_list)
        disp_train_R.extend(disp_R_list)
    # 加载测试数据
    test_path = os.path.join(filepath, "test")
    test_dir = os.listdir(test_path)

    left_test = []
    right_test = []
    disp_test_L = []
    disp_test_R = []

    for single_dir in test_dir:
        left_list, right_list, disp_L_list, disp_R_list = airsimfolderloader(os.path.join(test_path, single_dir))
        left_test.extend(left_list)
        right_test.extend(right_list)
        disp_test_L.extend(disp_L_list)
        disp_test_R.extend(disp_R_list)
    return left_train, right_train, disp_train_L, disp_train_R, left_test, right_test, disp_test_L, disp_test_R


if __name__ == "__main__":
    # a, b, c, d = airsimfolderloader("/home/immortalqx/data/datasets/airsim/train/2023-10-14-17-44-41")
    # print(d)
    a, b, c, d, e, f, g, h = listloader("/home/immortalqx/data/datasets/airsim")
    # print(a)
    # print(b)
    # print(c)
    # print(d)
    print(e)
    print(f)
    print(g)
    print(h)

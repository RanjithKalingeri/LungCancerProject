import pydicom as dicom
import os
import torch
import torch.nn.functional as nnf


mydevice = "cuda" if torch.cuda.is_available() else "cpu"
folder_path = r"C:\Users\ranji\Desktop\DeepLearning\manifest-cgqtDj7Y2699835271585651107\SPIE-AAPM Lung CT Challenge"
all_images = os.listdir(folder_path)
# print(all_images)
i = 1
for image in all_images[0:1]:
    first_click = os.path.join(folder_path, image)
    first_click_dir = os.listdir(first_click)
    second_click = os.path.join(first_click, first_click_dir[0])
    second_click_dir = os.listdir(second_click)
    third_click = os.path.join(second_click, second_click_dir[0])
    third_click_dir = os.listdir(third_click)
    # print(third_click_dir)
    lis = []
    for dimage in third_click_dir:
        ds = dicom.dcmread(os.path.join(third_click, dimage))
        pixel_array_numpy = ds.pixel_array
        pixel_array_numpy = pixel_array_numpy.astype("float64")
        d_tensor = torch.from_numpy(pixel_array_numpy)
        lis.append(d_tensor)
    final3d_tensor = torch.stack(lis, dim=0)


    final3d_tensor = final3d_tensor.permute(2, 1, 0)
    final3d_tensor = nnf.interpolate(final3d_tensor, size=250)
    # final3d_tensor = final3d_tensor.permute(2, 1, 0)
    print(final3d_tensor.shape)
    # nnf.normalize(final3d_tensor, p=2, dim=1)
    # print(final3d_tensor)
    # mean, std = final3d_tensor.mean([1,2]), final3d_tensor.std([1,2])
    # print(mean)
    # print(std)
    print("---------")
    i = i+1

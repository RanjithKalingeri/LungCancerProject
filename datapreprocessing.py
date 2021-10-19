import pydicom as dicom
import os
import torch
import torch.nn.functional as nnf


mydevice = "cuda" if torch.cuda.is_available() else "cpu"
folder_path = r"C:\Users\ranji\Desktop\DeepLearning\manifest-cgqtDj7Y2699835271585651107\SPIE-AAPM Lung CT Challenge"
all_images = os.listdir(folder_path)
# print(all_images)
k = 0
for image in all_images:
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
        # image_hfu = [[0]*512 for i in range(512)]
        updated_image = [[0] * 512 for i in range(512)]
        for i in range(512):
            for j in range(512):
                image_hfu = pixel_array_numpy[i][j] - float(1024)
                if image_hfu > float(-1000) and image_hfu < float(-300):
                    updated_image[i][j] = pixel_array_numpy[i][j]/float(255)
        image_tensor = torch.Tensor(updated_image)
        image_tensor.to(mydevice)
        lis.append(image_tensor)
    final3d_tensor = torch.stack(lis, dim=0)
    final3d_tensor.to(mydevice)
    final3d_tensor = final3d_tensor.permute(2, 1, 0)
    final3d_tensor = nnf.interpolate(final3d_tensor, size=250)
    # final3d_tensor = final3d_tensor.permute(2, 1, 0)
    file_name = r'C:\Users\ranji\Desktop\DeepLearning\patient3D_tensors'
    image_name = 'final3dimage' + str(k)
    final_path = os.path.join(file_name, image_name)
    # print(final_path)
    k = k+1
    torch.save(final3d_tensor, final_path)
    print(final3d_tensor.shape)
    print(k)
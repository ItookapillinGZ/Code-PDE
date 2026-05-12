import h5py
import matplotlib.pyplot as plt

file_path = 'ReacDiff_Nu0.5_Rho5.0_Clean100.hdf5'
#
# with h5py.File(file_path, 'r') as f:
#     # 取第 0 个样本，第 0 个时间步，所有的空间点
#     initial_condition = f['tensor'][20, 1, :]
#
#     # 取对应的空间坐标
#     x = f['x-coordinate'][:]
#
#     print(f"初值数据的形状: {initial_condition.shape}")
#     print("前 10 个空间点的初值数值：\n", initial_condition[:10])
#
# # 可视化：一眼就能看出是正弦波、高斯脉冲还是随机噪声
# plt.plot(x, initial_condition)
# plt.title("Initial Condition (t=0)")
# plt.xlabel("x")
# plt.ylabel("u(x, 0)")
# plt.grid(True)
# plt.show()
# # 替换为你的文件路径
# file_path = 'ReacDiff_Nu0.5_Rho1.0.hdf5'
#
# with h5py.File(file_path, 'r') as f:
#     # 打印文件中所有的 key (类似文件夹名或变量名)
#     print("文件中的基本结构:", list(f.keys()))
#
#     # 假设里面有一个叫 'data' 的数据集
#     if 'data' in f:
#         data = f['data'][:]  # 读取数据到内存
#         print("数据内容:\n", data)
#         print("数据形状:", data.shape)

#
# # file_path = 'ReacDiff_Nu0.5_Rho1.0.hdf5'
#
# with h5py.File(file_path, 'r') as f:
#     for key in f.keys():
#         obj = f[key]
#         if isinstance(obj, h5py.Dataset):
#             print(f"数据项: {key:15} | 形状: {obj.shape} | 类型: {obj.dtype}")

import h5py
import numpy as np

# 配置文件名
old_file = 'ReacDiff_Nu0.5_Rho5.0.hdf5'
new_file = 'ReacDiff_Nu0.5_Rho5.0_Clean100.hdf5'
num_samples_to_keep = 100

with h5py.File(old_file, 'r') as f_old:
    total_samples = f_old['tensor'].shape[0]

    # 1. 随机生成不重复的索引并排序（保证 h5py 提取效率）
    print(f"Generating random indices for {num_samples_to_keep} samples...")
    indices = np.sort(np.random.choice(total_samples, num_samples_to_keep, replace=False))

    with h5py.File(new_file, 'w') as f_new:
        # 2. 提取 tensor 数据
        # 形状预期从 (10000, 101, 1024) 变为 (100, 101, 1024)
        print(f"Creating dataset 'tensor' in {new_file}...")
        f_new.create_dataset('tensor', data=f_old['tensor'][indices, :, :], compression="gzip")

        # 3. 处理时间坐标 (t-coordinate)
        # 如果长度是 102，裁剪为 101 以匹配 tensor 的时间维度
        print("Processing and copying coordinates...")
        t_data = f_old['t-coordinate'][:]
        if len(t_data) == 102:
            print(f"Original t-coordinate length 102. Clipping to 101 to match tensor.")
            t_data = t_data[:101]

        f_new.create_dataset('t-coordinate', data=t_data)

        # 4. 复制空间坐标 (x-coordinate)
        f_new.create_dataset('x-coordinate', data=f_old['x-coordinate'][:])

        # 5. 关键步骤：更新元数据属性
        # 确保新文件中的属性反映了当前的物理参数
        print(f"Setting attributes: nu=0.5, rho=10.0")
        f_new.attrs['nu'] = 0.5
        f_new.attrs['rho'] = 10.0

        # 同时复制原文件中其他可能存在的元数据
        for attr_name, attr_value in f_old.attrs.items():
            if attr_name not in ['nu', 'rho']:
                f_new.attrs[attr_name] = attr_value

        # 6. 验证输出结构
        print(f"\n" + "=" * 30)
        print(f"SUCCESS: {new_file} has been created.")
        print(f"Final tensor shape: {f_new['tensor'].shape} (Samples, Time, Space)")
        print(f"Final t-coordinate shape: {f_new['t-coordinate'].shape}")
        print(f"Final t-interval (dt): {t_data[1] - t_data[0]:.4f}")
        print(f"=" * 30)
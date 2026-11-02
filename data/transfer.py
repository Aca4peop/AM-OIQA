import numpy as np
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor, as_completed
orient = ['A','B','C','D','E','F']
center = [(0, 0),(np.pi/3, 0),(2*np.pi/3, 0),(-np.pi/3, 0),(-2*np.pi/3, 0),(np.pi, 0)]
width = 512

def gen_xyz(fov, u, v, out_h, out_w):
    out = np.ones((out_h, out_w, 3), np.float32)

    x_rng = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), num=out_w, dtype=np.float32)
    y_rng = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), num=out_h, dtype=np.float32)

    out[:, :, :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)
    Rx = np.array([[1, 0, 0], [0, np.cos(v), -np.sin(v)], [0, np.sin(v), np.cos(v)]])
    Ry = np.array([[np.cos(u), 0, np.sin(u)], [0, 1, 0], [-np.sin(u), 0, np.cos(u)]])

    R = np.dot(Ry, Rx)
    return out.dot(R.T)

def xyz_to_uv(xyz):
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x ** 2 + z ** 2)
    v = np.arctan2(y, c)
    return np.concatenate([u, v], axis=-1)

def uv_to_XY(uv, eq_h, eq_w):
    u, v = np.split(uv, 2, axis=-1)
    X = (u / (2 * np.pi) + 0.5) * eq_w - 0.5
    Y = (-v / np.pi + 0.5) * eq_h - 0.5
    return np.concatenate([X, Y], axis=-1)

def eq_to_pers(eqimg, fov, u, v, out_h, out_w):
    # 生成三维坐标点
    xyz = gen_xyz(fov, u, v, out_h, out_w)
    # 将三维坐标点转换为二维平面坐标点
    uv  = xyz_to_uv(xyz)

    # 获取等距图像的高度和宽度
    eq_h, eq_w = eqimg.shape[:2]
    # 将二维平面坐标点转换为图像坐标点
    XY = uv_to_XY(uv, eq_h, eq_w)

    # 将图像坐标点分离为X和Y分量
    X, Y = np.split(XY, 2, axis=-1)
    # 将X分量重塑为指定的高度和宽度
    X = np.reshape(X, (out_h, out_w))
    # 将Y分量重塑为指定的高度和宽度
    Y = np.reshape(Y, (out_h, out_w))

    # 使用ndimage.map_coordinates函数对蓝色通道进行重采样
    mc0 = ndimage.map_coordinates(eqimg[:, :, 0], [Y, X]) # channel: B
    # 使用ndimage.map_coordinates函数对绿色通道进行重采样
    mc1 = ndimage.map_coordinates(eqimg[:, :, 1], [Y, X]) # channel: G
    # 使用ndimage.map_coordinates函数对红色通道进行重采样
    mc2 = ndimage.map_coordinates(eqimg[:, :, 2], [Y, X]) # channel: R

    # 将三个通道的图像数据堆叠成一个三维数组
    output = np.stack([mc0, mc1, mc2], axis=-1)
    return output


def single_image(args):
    input_path, filename = args
    try:
        img = Image.open(os.path.join(input_path, filename))
        img = np.array(img).astype(np.float64)
        name = filename.split('.')[0]

        images = [eq_to_pers(img, np.pi/4,center[t][0],center[t][1], width, width) for t in range(len(orient))]
        images = [Image.fromarray(image.astype(np.uint8)) for image in images]
        for t in range(len(orient)):
            images[t].save(os.path.join(output_path, f"{name}_{orient[t]}.png"))
        return filename
    except Exception as e:
        print(f"Process error {filename}: {e}")
        return None



if __name__ == "__main__":
    from tqdm import tqdm
    import os
    import PIL.Image as Image
    # input_path = '/home1/Databases/IQA-ODI/all_ref_test_img/'
    # input_path = '/home1/lyq/Dataset/OIQA/distorted_images/'
    #input_path = '/home1/Databases/CVIQ/CVIQ/'
    #input_path = '/home1/Databases/MVAQD/imgs/'
    #input_path = '/home1/Databases/JUFE/dis/'
    input_path   ='/home1/server823-2/database/OIQ-10K/'
    # output_path = '/home1/lyq/Dataset/IQA-ODI/resized_cubic_equator/'
    #output_path = '/home1/lyq/Dataset/OIQA/resized_cubic_equator/'
    #output_path = '/home1/lyq/Dataset/CVIQ/resized_cubic_equator/'
    #output_path = '/home1/lyq/Dataset/MVAQD/resized_cubic_equator/'
    #output_path = '/home1/lyq/Dataset/JUFE/resized_cubic_equator/'
    output_path ='/home1/lyq/dataset/OIQ/resized_cubic_equator/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_list = os.listdir(input_path)
    for filename in file_list:
        if not filename.endswith(('.jpg', '.png')):
            file_list.remove(filename)
    
    num_workers = 10
    total_tasks = len(file_list)
    assert total_tasks == 10000, "Img file incomplete."
    print(f"开始处理 {total_tasks} 张图像，使用 {num_workers} 个进程...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(single_image,(input_path, filename)): (input_path, filename) for filename in file_list}
        
        # 使用tqdm显示进度条
        with tqdm(total=total_tasks, desc="图像处理进度", unit="张") as pbar:
            for future in as_completed(future_to_path):
                try:
                    result = future.result()
                    if result:
                        pbar.set_postfix({"当前处理": os.path.basename(result)})
                except Exception as e:
                    print(f"任务异常: {e}")
                finally:
                    pbar.update(1)  # 更新进度条
            

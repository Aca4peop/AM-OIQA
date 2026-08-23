import argparse

import numpy as np
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor, as_completed
orient = ['A','B','C','D','E','F'] # Orentations of the 6 perspective views
center = [(0, 0),(np.pi/3, 0),(2*np.pi/3, 0),(-np.pi/3, 0),(-2*np.pi/3, 0),(np.pi, 0)]# Centers of the 6 perspective views
width = 512

def gen_xyz(fov, u, v, out_h, out_w):
    out = np.ones((out_h, out_w, 3), np.float32)  
    # Initialize output array with z=1

    x_rng = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), num=out_w, dtype=np.float32)  
    # Generate x range based on FOV
    y_rng = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), num=out_h, dtype=np.float32)  
    # Generate y range based on FOV

    out[:, :, :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)  
    # Create 2D grid of (x, y) coordinates
    Rx = np.array([[1, 0, 0], [0, np.cos(v), -np.sin(v)], [0, np.sin(v), np.cos(v)]])  
    # Rotation matrix around x-axis
    Ry = np.array([[np.cos(u), 0, np.sin(u)], [0, 1, 0], [-np.sin(u), 0, np.cos(u)]])  
    # Rotation matrix around y-axis

    R = np.dot(Ry, Rx)  
    # Combined rotation matrix
    return out.dot(R.T)  
    # Apply rotation and return rotated coordinates

def xyz_to_uv(xyz):
    x, y, z = np.split(xyz, 3, axis=-1)  
    # Split xyz coordinates into x, y, z components
    u = np.arctan2(x, z)  
    # Calculate azimuthal angle u (longitude)
    c = np.sqrt(x ** 2 + z ** 2)  
    # Calculate radial distance in xz plane
    v = np.arctan2(y, c)  
    # Calculate polar angle v (latitude)
    return np.concatenate([u, v], axis=-1)  
# Concatenate u and v into spherical coordinates

def uv_to_XY(uv, eq_h, eq_w):
    u, v = np.split(uv, 2, axis=-1)  
    # Split spherical coordinates into u and v
    X = (u / (2 * np.pi) + 0.5) * eq_w - 0.5  
    # Convert u to pixel X coordinate
    Y = (-v / np.pi + 0.5) * eq_h - 0.5  
    # Convert v to pixel Y coordinate
    return np.concatenate([X, Y], axis=-1)  
# Concatenate X and Y coordinates

def eq_to_pers(eqimg, fov, u, v, out_h, out_w):
    xyz = gen_xyz(fov, u, v, out_h, out_w)  
    # Generate 3D coordinates for perspective view
    uv  = xyz_to_uv(xyz)  
    # Convert xyz to spherical coordinates

    eq_h, eq_w = eqimg.shape[:2]  
    # Get equirectangular image dimensions
    XY = uv_to_XY(uv, eq_h, eq_w)  
    # Convert spherical coordinates to pixel coordinates

    X, Y = np.split(XY, 2, axis=-1)  
    # Split X and Y coordinates
    X = np.reshape(X, (out_h, out_w))  
    # Reshape X to output image size
    Y = np.reshape(Y, (out_h, out_w))  
    # Reshape Y to output image size

    mc0 = ndimage.map_coordinates(eqimg[:, :, 0], [Y, X]) # channel: B
    mc1 = ndimage.map_coordinates(eqimg[:, :, 1], [Y, X]) # channel: G
    mc2 = ndimage.map_coordinates(eqimg[:, :, 2], [Y, X]) # channel: R


    output = np.stack([mc0, mc1, mc2], axis=-1)  
    # Stack BGR channels into output image
    return output


def single_image(args):
    input_path, filename = args  
    # Unpack input path and filename
    try:
        img = Image.open(os.path.join(input_path, filename))  
        # Load image from disk
        img = np.array(img).astype(np.float64)  
        # Convert to numpy array with float64 precision
        name = filename.split('.')[0]  
        # Extract filename without extension

        images = [eq_to_pers(img, np.pi/4,center[t][0],center[t][1], width, width) for t in range(len(orient))]  
        # Convert equirectangular to 6 perspective views
        images = [Image.fromarray(image.astype(np.uint8)) for image in images]  
        # Convert arrays back to PIL images
        for t in range(len(orient)):
            images[t].save(os.path.join(output_path, f"{name}_{orient[t]}.png"))  
            # Save each perspective view with orientation label
        return filename
    except Exception as e:
        print(f"Process error {filename}: {e}")  
        # Print error message if processing fails
        return None
def paser():
    parser = argparse.ArgumentParser(description='Process some integers.')  
    # Create argument parser
    parser.add_argument('--input_path', type=str, default='/home1/server823-2/database/OIQ-10K/', help='input path of the dataset')  
    # Input dataset path
    parser.add_argument('--output_path', type=str, default='', help='output path of the dataset')  
    # Output dataset path
    args = parser.parse_args()  
    # Parse command line arguments
    return args


if __name__ == "__main__":
    from tqdm import tqdm
    import os
    import PIL.Image as Image
    args = paser()
    input_path  = args.input_path  
    # Get input path from arguments
    output_path = args.output_path  
    # Get output path from arguments
    if not os.path.exists(output_path):
        os.makedirs(output_path)  
        # Create output directory if not exists
    file_list = os.listdir(input_path)  
    # List all files in input directory
    for filename in file_list:
        if not filename.endswith(('.jpg', '.png')):
            file_list.remove(filename)  
            # Filter only image files
    num_workers = 10  
    # Number of parallel processes
    total_tasks = len(file_list)
    assert total_tasks == 10000, "Img file incomplete."  
    # Verify dataset completeness
    print(f"开始处理 {total_tasks} 张图像，使用 {num_workers} 个进程...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(single_image,(input_path, filename)): (input_path, filename) for filename in file_list}  # Submit all tasks
        with tqdm(total=total_tasks, desc="图像处理进度", unit="张") as pbar:
            for future in as_completed(future_to_path):  
                # Process completed tasks
                try:
                    result = future.result()
                    if result:
                        pbar.set_postfix({"当前处理": os.path.basename(result)})  
                        # Update progress bar with current file
                except Exception as e:
                    print(f"任务异常: {e}")
                finally:
                    pbar.update(1)  
                    # Increment progress bar  
            

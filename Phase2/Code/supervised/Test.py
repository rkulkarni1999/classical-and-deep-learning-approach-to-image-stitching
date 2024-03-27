# import torch
# import cv2
# import numpy as np
# import torchvision.transforms as transforms
# from Network.Network import HomographyModel  # Adjust this import according to your project structure
# import argparse
# import os

# def prepare_input_tensor(image1, image2, image_size=(128, 128)):
#     """
#     Prepare an input tensor for the model from two image paths.
#     """
#     # Load images in grayscale
#     print(image1.shape)
#     # cv2.imshow("image", image1)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
#     image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

#     if image1 is None or image2 is None:
#         raise ValueError("One or both images could not be loaded.")

#     # Resize images
#     # image1 = cv2.resize(image1, image_size)
#     # image2 = cv2.resize(image2, image_size)

#     # Convert to PyTorch tensors
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#     transforms.CenterCrop((128, 128)),  # Center crop to 128x128
#     transforms.ToTensor()  # Convert to tensor
# ])
#     image1_tensor = transform(image1)
#     image2_tensor = transform(image2)


#     # print(f"Image 1 : {image1_tensor}")
#     # print(f"Image 2 : {image1_tensor}")

#     # Stack tensors
#     input_tensor = torch.stack([image1_tensor, image2_tensor], dim=1)

#     return input_tensor

# def convert_to_homography_matrix(h_points):
#     """
#     Convert the predicted points to a homography matrix.
#     """
#     h_points = h_points.reshape(4, 2)
#     pts_src = np.array([[0, 0], [127, 0], [127, 127], [0, 127]], dtype=np.float32)
#     pts_dst = h_points + pts_src
#     homography_matrix, _ = cv2.findHomography(pts_src, pts_dst, method=0)
#     return homography_matrix


# def warpTwoImages(img1,img2,H):
#     h1,w1 = img1.shape[:2]
#     h2,w2 = img2.shape[:2]
#     pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
#     pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
#     pts2_= cv2.perspectiveTransform(pts2,H)
#     pts =np.concatenate((pts1,pts2_), axis=0)
#     [xmin,ymin] = np.int32(pts.min(axis=0).ravel())
#     [xmax,ymax] = np.int32(pts.max(axis=0).ravel())
#     t = [-xmin,-ymin]
#     Ht= np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
#     # translate
#     result = cv2.warpPerspective(img2,Ht.dot(H),(xmax-xmin,ymax-ymin),flags= cv2.INTER_LINEAR)
#     result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
#     return result

# def forward_pass(model, input_tensor):
#     """
#     Perform a forward pass through the model.
#     """
#     model.eval()
#     with torch.no_grad():
#         output = model(input_tensor)
#     return output

# def load_checkpoint(model, checkpoint_path):
#     """
#     Load a model checkpoint.
#     """
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint['model_state_dict'])
#     return model

# def main(image1, image2, checkpoint_path):
#     """
#     Main function to run the end-to-end process.
#     """
#     # Initialize the model
#     model = HomographyModel()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load the model checkpoint
#     model = load_checkpoint(model, checkpoint_path)
#     model = model.to(device)

#     # Prepare input tensor
#     input_tensor = prepare_input_tensor(image1, image2)
#     input_tensor = input_tensor.to(device)

#     # Forward pass
#     output = forward_pass(model, input_tensor)

#     # Convert output to homography matrix
#     h_points_four = output.view(-1).cpu().numpy()

#     print(f"H_points_4 : {h_points_four}")
#     homography_matrix = convert_to_homography_matrix(h_points_four)

#     # Load the original images in color
#     # image1 = cv2.imread(image_path1)
#     # image2 = cv2.imread(image_path2)

#     # Blend the images
#     blended_image = warpTwoImages(image2, image1, homography_matrix)

#     # Display or save the blended image
#     # cv2.imshow("Blended Image", blended_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     return blended_image

# if __name__ == "__main__":
#     # image_path1 = '/home/pear_group/rutwik/computer_vision_hw/rkulkarni1/Phase2/Data/Train/Train_original_multiple/image_1.png'  # Replace with your image path
#     # image_path2 = '/home/pear_group/rutwik/computer_vision_hw/rkulkarni1/Phase2/Data/Train/Train_warped_multiple/image_1.png' # Replace with your image path
#     Parser = argparse.ArgumentParser()
#     Parser.add_argument('--ImagePath', default="/home/pear_group/rutwik/computer_vision_hw/rkulkarni1/Phase1/Data/Train/Set3", help= 'path to image directory')
#     Args = Parser.parse_args()
#     image_path = Args.ImagePath
#     if os.path.exists(image_path):
#         image_list = os.listdir(image_path)
#         image_list.sort()
#     else:
#         raise Exception("Incorrect pATH")
    
#     image_path1 = os.path.join(image_path,image_list[0])
#     image1 = cv2.imread(image_path1)
#     checkpoint_path = '/home/pear_group/rutwik/computer_vision_hw/rkulkarni1/Phase2/supervised/Checkpoints/198model.ckpt'

#     for i in range(1,len(image_list)):
#         image_path2 = os.path.join(image_path,image_list[i])
#         image2 = cv2.imread(image_path2)
#         blended_image = main(image1, image2, checkpoint_path)
#         cv2.imwrite("pano.png",blended_image)
#         image1 = blended_image
#         print(i)
    
#       # Replace with your checkpoint path

import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from Network.Network import HomographyModel  # Adjust this import according to your project structure
import argparse
import os
from kornia.geometry.transform.imgwarp import get_perspective_transform
from PIL import Image

def prepare_input_tensor(image1, image2, image_size=(128, 128)):
    """
    Prepare an input tensor for the model from two image paths.
    """
    # Load images in grayscale
    print(image1.shape)
    # cv2.imshow("image", image1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    if image1 is None or image2 is None:
        raise ValueError("One or both images could not be loaded.")

    # Resize images
    # image1 = cv2.resize(image1, image_size)
    # image2 = cv2.resize(image2, image_size)

    # Convert to PyTorch tensors
    transform = transforms.Compose([
        transforms.ToPILImage(),
    transforms.CenterCrop((128, 128)),  # Center crop to 128x128
    transforms.ToTensor()  # Convert to tensor
])
    image1_tensor = transform(image1)
    image2_tensor = transform(image2)
    image1_np = image1_tensor.numpy().transpose(1, 2, 0)
    image2_np = image2_tensor.numpy().transpose(1, 2, 0)
    # Scale to [0, 255]
    image_np1 = (image1_np * 255).astype(np.uint8)
    image_np2 = (image2_np * 255).astype(np.uint8)
    # Display the image
    # cv2.imshow('Image', image_np1)
    # cv2.imshow('Image2', image_np2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(f"Image 1 : {image1_tensor}")
    # print(f"Image 2 : {image1_tensor}")

    # Stack tensors
    input_tensor = torch.stack([image1_tensor, image2_tensor], dim=1)

    return input_tensor, image_np1, image_np2

def convert_to_homography_matrix(h_points):
    """
    Convert the predicted points to a homography matrix.
    """
    h_points = h_points.reshape(4, 2)
    pts_src = np.array([[0, 0], [127, 0], [127, 127], [0, 127]], dtype=np.float32)
    pts_dst = h_points + pts_src
    homography_matrix, _ = cv2.findHomography(pts_src, pts_dst, method=0)
    return homography_matrix


def warpTwoImages(img1,img2,H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_= cv2.perspectiveTransform(pts2,H)
    pts =np.concatenate((pts1,pts2_), axis=0)
    [xmin,ymin] = np.int32(pts.min(axis=0).ravel())
    [xmax,ymax] = np.int32(pts.max(axis=0).ravel())
    t = [-xmin,-ymin]
    Ht= np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    # translate
    result = cv2.warpPerspective(img2,Ht.dot(H),(xmax-xmin,ymax-ymin),flags= cv2.INTER_LINEAR)
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

def create_panorama(image1, image2, homography_matrix):
    # Warping (transforming) image2 with the given homography matrix
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Calculate the size of the panorama
    corners_image2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_image2, homography_matrix)
    
    [x_min, y_min] = np.int32(transformed_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(transformed_corners.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_size = (x_max - x_min, y_max - y_min)
    warped_image2 = cv2.warpPerspective(image2, H_translation.dot(homography_matrix), output_size)
    warped_image2[translation_dist[1]:height1+translation_dist[1], translation_dist[0]:width1+translation_dist[0]] = image1

    return warped_image2

def forward_pass(model, input_tensor):
    """
    Perform a forward pass through the model.
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output

def load_checkpoint(model, checkpoint_path):
    """
    Load a model checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main(pano_image, image1, image2, checkpoint_path):
    """
    Main function to run the end-to-end process.
    """
    # Initialize the model
    model = HomographyModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model checkpoint
    model = load_checkpoint(model, checkpoint_path)
    model = model.to(device)

    # Prepare input tensor
    input_tensor,image_np1, image_np2= prepare_input_tensor(image1, image2)
    input_tensor = input_tensor.to(device)

    # Forward pass
    output = forward_pass(model, input_tensor)

    # Convert output to homography matrix
    h_points_four = output.view(-1).cpu().numpy()

    print(f"H_points_4 : {h_points_four}")
    homography_matrix = convert_to_homography_matrix(h_points_four)
    H_warp = np.rint(np.linalg.inv(homography_matrix))
    H_warp[0, -1] *= -3
    H_warp[1, -1] *= -3
    # if H_warp[1,2] > 0:
    #     H_warp = -1 * H_warp[1,2]
    print(H_warp)

    # H_inv = np.linalg.inv(homography_matrix)
    # warped_img = cv2.warpPerspective(image1, H_inv, (128, 128))
    # Load the original images in color
    # image1 = cv2.imread(image_path1)
    # image2 = cv2.imread(image_path2)

    # Blend the images
    # image1 = cv2.resize(image1, (480, 270))
    # image2 = cv2.resize(image2, (480, 270))
    flipped_image2 = cv2.flip(image2, 1)
    flipped_pano = cv2.flip(pano_image, 1)
    flipped_blended_image = warpTwoImages( flipped_image2, flipped_pano,  H_warp)
    blended_image = cv2.flip(flipped_blended_image, 1)
    # blended_image = create_panorama(image2, pano_image, H_warp)

    # blended_image = warpTwoImages( image2, pano_image,  H_warp)

    # Display or save the blended image
    # cv2.imshow("Blended Image", blended_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return blended_image

if __name__ == "__main__":
    # image_path1 = '/home/pear_group/rutwik/computer_vision_hw/rkulkarni1/Phase2/Data/Train/Train_original_multiple/image_1.png'  # Replace with your image path
    # image_path2 = '/home/pear_group/rutwik/computer_vision_hw/rkulkarni1/Phase2/Data/Train/Train_warped_multiple/image_1.png' # Replace with your image path
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ImagePath', default="D:\Study\RBE 549\Project1\computer_vision_hw\computer_vision_hw\\rkulkarni1\Phase2\supervisedtest", help= 'path to image directory')
    Args = Parser.parse_args()
    image_path = Args.ImagePath
    if os.path.exists(image_path):
        file_list = os.listdir(image_path)
        image_list =sorted(file_list, key=lambda x: int(x.split('.')[0]))
        print(image_list)
        # fgvu
    else:
        raise Exception("Incorrect pATH")
    
    image_path1 = os.path.join(image_path,image_list[0])
    image1 = cv2.imread(image_path1)
    pano_image = image1
    # image1 = cv2.resize(image1, (480, 270))
    checkpoint_path =  r'D:\Study\RBE 549\Project1\computer_vision_hw\computer_vision_hw\rkulkarni1\Phase2\supervised\Checkpoints_trans\\40model.ckpt'

    for i in range(0,len(image_list)-1):
        image_path1 = os.path.join(image_path,image_list[i])
        image1 = cv2.imread(image_path1)
        image_path2 = os.path.join(image_path,image_list[i+1])
        image2 = cv2.imread(image_path2)
        # pano_image = image1
        # image2 = cv2.resize(image2, (480, 270))
        blended_image = main(pano_image,image1, image2, checkpoint_path)
        cv2.imwrite("pano.png",blended_image)
        pano_image = blended_image
        print(i)
    
      # Replace with your checkpoint path
    


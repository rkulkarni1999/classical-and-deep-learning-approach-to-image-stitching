import torch
import argparse
import kornia
import imageio
import numpy as np
import cv2 

from Network.UN_Network import HomographyModel
from Dataset.dataset import SyntheticDataset
from kornia.geometry.transform.imgwarp import get_perspective_transform, warp_perspective

def tensors_to_gif(a, b, name):
    a = a.permute(1, 2, 0).numpy()
    b = b.permute(1, 2, 0).numpy()
    imageio.mimsave(name, [a, b], duration=1)

def load_checkpoint(model, checkpoint_path):
    """
    Load a model checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

@torch.no_grad()
def main(args):

    model = HomographyModel(model_hparams=Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(device)
    # Load the model checkpoint
    model = load_checkpoint(model, checkpoint_path)
    model = model.to(device)

    # model = HomographyModel.load_from_checkpoint(args.checkpoint)
    model.eval()
    test_set = SyntheticDataset(args.test_path, rho=args.rho)

    for i in range(args.n):
        img_a,img_b, patch_a, patch_b, corners, delta = test_set[i]
        # print(type(patch_a))
        # tensors_to_gif(patch_a, patch_b, f"figures/input_{i}.gif")
        image1_np = img_a.numpy().transpose(1, 2, 0)
        image2_np = img_b.numpy().transpose(1, 2, 0)
        # Scale to [0, 255]
        image_np1 = (image1_np * 255).astype(np.uint8)
        image_np2 = (image2_np * 255).astype(np.uint8)
    # Display the image
    # cv2.imshow('Image', image_np1)
    # cv2.imshow('Image2', image_np2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        cv2.imshow("patch_a",  image_np1)
  
        cv2.imshow("patch_b",image_np2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("image11.jpg", image_np1)
        cv2.imwrite("image12.jpg", image_np2)

        patch_a = patch_a.unsqueeze(0)
        patch_b = patch_b.unsqueeze(0)
        corners = corners.unsqueeze(0)
        print(patch_a.shape)
        print(patch_b.shape)
        corners = corners - corners[:, 0].view(-1, 1, 2)

        delta_hat = model(patch_a, patch_b)
        corners_hat = corners + delta_hat
        h = get_perspective_transform(corners, corners_hat)
        h_inv = torch.inverse(h)
        # print(patch_a.shape)
        img_a = torch.unsqueeze(img_a, 0)
        # print(img_a.shape)
        patch_b_hat = warp_perspective(img_a, h_inv, (256, 256))
        # tensors_to_gif(patch_b_hat[0], patch_b[0], f"figures/output_{i}.gif")
        patch_b_hat = patch_b_hat.squeeze()

        patch_b_hat = patch_b_hat.numpy()
        # Scale to [0, 255]
        patch_b_hat = (patch_b_hat * 255).astype(np.uint8)
        cv2.imshow("patch_a", patch_b_hat)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("predicted1.jpg", patch_b_hat)

if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--train_path",
        default="D:\Study\RBE 549\Project1\computer_vision_hw\computer_vision_hw\\rkulkarni1\Phase2\Data\Train\Train",
        help="Base path of images",
    )
    Parser.add_argument(
        "--test_path",
        default="D:\Study\RBE 549\Project1\computer_vision_hw\computer_vision_hw\\rkulkarni1\Phase2\Data\Val\Val",
        help="Base path of images",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpointss/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=1,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )
    Parser.add_argument(
        "--rho", 
        type=int, 
        default=45, help="amount to perturb corners")

    Parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="learning rate")
    
    Parser.add_argument("--n", type=int, default=5, help="number of images to test")
    Args = Parser.parse_args()
    checkpoint_path = r'D:\Study\RBE 549\Project1\computer_vision_hw\computer_vision_hw\rkulkarni1\Phase2\unsupervised\Code_unsupervised\Checkpointss\280model.ckpt'
    main(Args)
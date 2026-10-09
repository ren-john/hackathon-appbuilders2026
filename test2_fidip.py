import sys
import cv2
import numpy as np
PAIRS = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]

class FiDIP:
    def __init__(self, args):
        import torch
        sys.path.insert(0, str(args.repo / 'lib'))
        from config import cfg
        from models.adaptive_pose_hrnet import PoseHighResolutionNet
        from utils.transforms import get_affine_transform, flip_back
        from core.inference import get_final_preds
        self.torch = torch
        self.affine, self.flip_back, self.decode = get_affine_transform, flip_back, get_final_preds
        self.cfg = cfg.clone()
        self.cfg.merge_from_file(str(args.config or args.repo / 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'))
        self.size = tuple(map(int, self.cfg.MODEL.IMAGE_SIZE))
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        self.device = torch.device(device)
        self.model = PoseHighResolutionNet(self.cfg)
        self.model.load_state_dict(torch.load(args.model or args.repo / 'models/hrnet_fidip.pth',
                                             map_location='cpu', weights_only=True), strict=True)
        self.model.to(self.device).eval()
        self.flip = args.flip_test
        self.mean = torch.tensor([.485,.456,.406], device=self.device)[None,:,None,None]
        self.std = torch.tensor([.229,.224,.225], device=self.device)[None,:,None,None]
        print(f'FiDIP loaded on {self.device}', flush=True)

    def predict(self, rgb):
        torch = self.torch
        h, w = rgb.shape[:2]
        center = np.array([(w-1)/2, (h-1)/2], dtype=np.float32)
        aspect = self.size[0] / self.size[1]
        scale = np.array([max(w,h*aspect), max(h,w/aspect)], dtype=np.float32)/200*1.25
        crop = cv2.warpAffine(rgb, self.affine(center,scale,0,self.size), self.size)
        x = torch.from_numpy(crop.copy()).permute(2,0,1)[None].to(self.device,dtype=torch.float32)/255
        x = (x-self.mean)/self.std
        with torch.inference_mode():
            _, heatmaps = self.model(x)
            hm = heatmaps.cpu().numpy()
            if self.flip:
                _, flipped = self.model(torch.flip(x,[3]))
                hm = (hm+self.flip_back(flipped.cpu().numpy(),PAIRS))/2
        if hm.shape[1] != 17 or not np.isfinite(hm).all():
            raise RuntimeError('FiDIP did not produce 17 finite heatmaps')
        points, scores = self.decode(self.cfg,hm,center[None],scale[None])
        return points[0]/np.array([w,h]), scores[0,:,0]

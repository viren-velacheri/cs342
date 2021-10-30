import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    max_pool_output = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, stride=1, padding = max_pool_ks // 2)[0, 0]

    # tip: visualize is_peak and heatmap side by side.
    peak_finder = (max_pool_output <= heatmap).float()

    y_coord = torch.nonzero(peak_finder==1, as_tuple=True)[0]
    x_coord = torch.nonzero(peak_finder==1, as_tuple=True)[1]

    max_det = min(max_det, heatmap[peak_finder == 1].shape[0])
    peak_scores = torch.topk(heatmap[peak_finder == 1], max_det)[0]
    peak_indices = list(torch.topk(heatmap[peak_finder == 1], max_det)[1])

    peaks_list = list()
    i = 0
    while i in range(len(peak_indices)):
      peak_score = peak_scores[i]
      if (min_score < peak_score):
        peak_index = peak_indices[i]
        peaks_tuple = (peak_score, x_coord[peak_index], y_coord[peak_index])
        peaks_list.append(peaks_tuple)
      i += 1
    
    return peaks_list


class Detector(torch.nn.Module):
    def __init__(self):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        input_channels = 3
        num_classes = 6
        layers=[32,64,128]
        self.layers = torch.nn.Sequential(
            self.__block(input_channels, 32, (7, 1, 3)),
            self.__block(32, 64),
            self.__block(64, 128),
            torch.nn.AvgPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(2048, num_classes)
        )

    def __block(self, in_dim, out_dim, extra=(3, 1, 1), pool=(2, 2)):
      return torch.nn.Sequential(
          torch.nn.Conv2d(in_dim, out_dim, *extra),
          torch.nn.BatchNorm2d(out_dim),
          torch.nn.SiLU(),
          torch.nn.MaxPool2d(*pool),
      )


    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        return self.layers(x)

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        detect_heatmap = self.forward(image)
        detect_peak_1 = extract_peak(detect_heatmap[0][0], max_pool_ks=7, min_score=-5, max_det=30)
        detect_peak_2 = extract_peak(detect_heatmap[0][1], max_pool_ks=7, min_score=-5, max_det=30)
        detect_peak_3 = extract_peak(detect_heatmap[0][2], max_pool_ks=7, min_score=-5, max_det=30)
        
        detect_list_1 = []
        for peak in detect_peak_1:
            score = peak[0]
            cx = peak[1]
            cy = peak[2]
            detect_tuple = (score, int(cx), int(cy), 0, 0)
            detect_list_1.append(detect_tuple)

        detect_list_2 = []
        for peak in detect_peak_2:
            score = peak[0]
            cx = peak[1]
            cy = peak[2]
            detect_tuple = (score, int(cx), int(cy), 0, 0)
            detect_list_2.append(detect_tuple)

        detect_list_3 = []
        for peak in detect_peak_3:
            score = peak[0]
            cx = peak[1]
            cy = peak[2]
            detect_tuple = (score, int(cx), int(cy), 0, 0)
            detect_list_3.append(detect_tuple)
        
        return detect_list_1, detect_list_2, detect_list_3

def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()

from typing import TypedDict

import torch

from easy_local_features.submodules.git_raco import RaCo

from easy_local_features.feature.basemodel import BaseExtractor, MethodType
from easy_local_features.utils.ops import prepareImage


class RaCoConfig(TypedDict):
    max_num_keypoints: int
    nms_radius: int
    subpixel_sampling: bool
    subpixel_temp: float
    ranker: bool
    covariance_estimator: bool
    sort_by_ranker: bool
    resize: int


class RACO_baseline(BaseExtractor):
    """RaCo: Ranking and Covariance for Practical Learned Keypoints (cvg/RaCo).

    Detector-only method. Besides keypoints, it predicts per-keypoint
    detection scores, ranking scores (matching reliability), and 2x2
    covariance matrices (spatial uncertainty), available via return_dict=True.
    """

    METHOD_TYPE = MethodType.DETECTOR_ONLY
    default_conf = RaCoConfig(
        max_num_keypoints=2048,
        nms_radius=3,
        subpixel_sampling=True,
        subpixel_temp=0.5,
        ranker=True,
        covariance_estimator=True,
        sort_by_ranker=False,
        resize=None,
    )

    def __init__(self, conf={}):
        self.conf = conf
        self.resize = conf.get("resize", self.default_conf["resize"])

        self.DEV = torch.device("cpu")
        self.detector = RaCo(
            max_num_keypoints=conf.get("max_num_keypoints", self.default_conf["max_num_keypoints"]),
            nms_radius=conf.get("nms_radius", self.default_conf["nms_radius"]),
            subpixel_sampling=conf.get("subpixel_sampling", self.default_conf["subpixel_sampling"]),
            subpixel_temp=conf.get("subpixel_temp", self.default_conf["subpixel_temp"]),
            ranker=conf.get("ranker", self.default_conf["ranker"]),
            covariance_estimator=conf.get("covariance_estimator", self.default_conf["covariance_estimator"]),
            sort_by_ranker=conf.get("sort_by_ranker", self.default_conf["sort_by_ranker"]),
        )
        self.detector.eval()
        self.detector.to(self.DEV)
        # Intentionally no matcher: RaCo is detector-only.

    @torch.inference_mode()
    def detect(self, image, return_dict: bool = False):
        # RaCo normalizes (ImageNet) internally, so feed raw [0, 1] RGB.
        img = prepareImage(image, batch=True).to(self.DEV)

        out = self.detector.extract(img, resize=self.resize)

        if return_dict:
            return {
                "mkpts": out["keypoints"],
                "scores": out["keypoint_scores"],
                "ranker_scores": out.get("ranker_scores"),
                "covariances": out.get("covariances"),
            }
        return out["keypoints"]

    def detectAndCompute(self, image, return_dict: bool = False):
        raise NotImplementedError("RACO_baseline is detector-only; use detect(image).")

    def compute(self, image, keypoints):
        raise NotImplementedError("RACO_baseline is detector-only; it does not compute descriptors.")

    def match(self, image1, image2):
        raise NotImplementedError("RACO_baseline is detector-only; it does not support matching.")

    @property
    def has_detector(self):
        return True

    def to(self, device):
        self.detector.to(device)
        self.DEV = torch.device(device) if isinstance(device, str) else device
        return self


if __name__ == "__main__":
    from easy_local_features.utils import io, vis

    detector = RACO_baseline({"max_num_keypoints": 512})

    img0 = io.fromPath("tests/assets/megadepth0.jpg")
    img1 = io.fromPath("tests/assets/megadepth1.jpg")

    kps0 = detector.detect(img0)
    kps1 = detector.detect(img1)

    vis.plot_pair(img0, img1)
    vis.plot_keypoints(keypoints0=kps0.cpu(), keypoints1=kps1.cpu())
    vis.show()

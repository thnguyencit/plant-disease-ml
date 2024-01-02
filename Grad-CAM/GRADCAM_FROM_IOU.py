# Imports for loading learner and the GradCAM class IoU

import cv2
from cv2 import imshow as cv2_imshow
import matplotlib.pyplot as plt
import os
import numpy as np
from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
import scipy.ndimage


class BBoxerwGradCAM:
    def __init__(
        self, learner, heatmap, image_path, resize_scale_list, bbox_scale_list
    ):
        self.learner = learner
        self.heatmap = heatmap
        self.image_path = image_path
        self.resize_list = resize_scale_list
        self.scale_list = bbox_scale_list

        self.og_img, self.smooth_heatmap = self.heatmap_smoothing()

        (
            self.bbox_coords,
            self.poly_coords,
            self.grey_img,
            self.contours,
        ) = self.form_bboxes()

    def heatmap_smoothing(self):
        og_img = cv2.imread(self.image_path)
        heatmap = cv2.resize(
            self.heatmap, (self.resize_list[0], self.resize_list[1])
        )  # Resizing
        og_img = cv2.resize(
            og_img, (self.resize_list[0], self.resize_list[1])
        )  # Resizing
        """
        The minimum pixel value will be mapped to the minimum output value (alpha - 0)
        The maximum pixel value will be mapped to the maximum output value (beta - 155)
        Linear scaling is applied to everything in between.
        These values were chosen with trial and error using COLORMAP_JET to deliver the best pixel saturation for forming contours.
        """
        heatmapshow = cv2.normalize(
            heatmap, None, alpha=0, beta=155, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

        return og_img, heatmapshow

    def show_smoothheatmap(self):
        cv2_imshow(self.smooth_heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_bboxrectangle(self):
        cv2.rectangle(
            self.og_img,
            (self.bbox_coords[0], self.bbox_coords[1]),
            (
                self.bbox_coords[0] + self.bbox_coords[2],
                self.bbox_coords[1] + self.bbox_coords[3],
            ),
            (0, 0, 0),
            3,
        )
        cv2_imshow(self.og_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_contouredheatmap(self):
        img_col = cv2.merge([self.grey_img, self.grey_img, self.grey_img])
        cv2.fillPoly(img_col, self.contours, [36, 255, 12])
        cv2_imshow(img_col)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_bboxpolygon(self):
        cv2.polylines(self.og_img, self.poly_coords, True, (0, 0, 0), 2)
        cv2_imshow(self.og_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def form_bboxes(self):
        grey_img = cv2.cvtColor(self.smooth_heatmap, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grey_img, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)

        for item in range(len(contours)):
            cnt = contours[item]
            if len(cnt) > 20:
                x, y, w, h = cv2.boundingRect(cnt)
                poly_coords = [cnt]
                x = int(x * self.scale_list[0])
                y = int(y * self.scale_list[1])
                w = int(w * self.scale_list[2])
                h = int(h * self.scale_list[3])
                return [x, y, w, h], poly_coords, grey_img, contours
            else:
                print("contour error (too small)")

    def get_bboxes(self):
        return self.bbox_coords, self.poly_coords


class GradCam:
    @classmethod
    def from_interp(
        cls, learn, interp, img_idx, ds_type=DatasetType.Test, include_label=False
    ):
        
        if ds_type == DatasetType.Test:
            ds = interp.data.valid_ds
        elif ds_type == DatasetType.Test:
            ds = interp.data.test_ds
            include_label = False
        else:
            return None

        x_img = ds.x[img_idx]
        xb, _ = interp.data.one_item(x_img)
        xb_img = Image(interp.data.denorm(xb)[0])
        probs = interp.preds[img_idx].numpy()

        pred_idx = interp.pred_class[
            img_idx
        ].item()  
        hmap_pred, xb_grad_pred = get_grad_heatmap(
            learn, xb, pred_idx, size=xb_img.shape[-1]
        )
        prob_pred = probs[pred_idx]

        actual_args = None
        if include_label:
            actual_idx = ds.y.items[img_idx]  
            if actual_idx != pred_idx:
                hmap_actual, xb_grad_actual = get_grad_heatmap(
                    learn, xb, actual_idx, size=xb_img.shape[-1]
                )
                prob_actual = probs[actual_idx]
                actual_args = [
                    interp.data.classes[actual_idx],
                    prob_actual,
                    hmap_actual,
                    xb_grad_actual,
                ]

        return cls(
            xb_img,
            interp.data.classes[pred_idx],
            prob_pred,
            hmap_pred,
            xb_grad_pred,
            actual_args,
        )

    @classmethod
    def from_one_img(cls, learn, x_img, label1=None, label2=None):
        """
        learn: fastai's Learner
        x_img: fastai.vision.image.Image
        label1: generate heatmap according to this label. If None, this wil be the label with highest probability from the model
        label2: generate additional heatmap according to this label
        """
        pred_class, pred_idx, probs = learn.predict(x_img)
        label1 = str(pred_class) if not label1 else label1

        xb, _ = learn.data.one_item(x_img)
        xb_img = Image(learn.data.denorm(xb)[0])
        probs = probs.numpy()

        label1_idx = learn.data.classes.index(label1)
        hmap1, xb_grad1 = get_grad_heatmap(learn, xb, label1_idx, size=xb_img.shape[-1])
        prob1 = probs[label1_idx]

        label2_args = None
        if label2:
            label2_idx = learn.data.classes.index(label2)
            hmap2, xb_grad2 = get_grad_heatmap(
                learn, xb, label2_idx, size=xb_img.shape[-1]
            )
            prob2 = probs[label2_idx]
            label2_args = [label2, prob2, hmap2, xb_grad2]

        return cls(xb_img, label1, prob1, hmap1, xb_grad1, label2_args)

    def __init__(self, xb_img, label1, prob1, hmap1, xb_grad1, label2_args=None):
        self.xb_img = xb_img
        self.label1, self.prob1, self.hmap1, self.xb_grad1 = (
            label1,
            prob1,
            hmap1,
            xb_grad1,
        )
        if label2_args:
            self.label2, self.prob2, self.hmap2, self.xb_grad2 = label2_args

    def plot(self, plot_hm=True, plot_gbp=True):
        if not plot_hm and not plot_gbp:
            plot_hm = True
        cols = 5 if hasattr(self, "label2") else 3
        if not plot_gbp or not plot_hm:
            cols -= 2 if hasattr(self, "label2") else 1

        fig, row_axes = plt.subplots(1, cols, figsize=(cols * 5, 5))
        col = 0
        size = self.xb_img.shape[-1]
        self.xb_img.show(row_axes[col])
        col += 1

        label1_title = f"1.{self.label1} {self.prob1:.3f}"
        if plot_hm:
            show_heatmap(self.hmap1, self.xb_img, size, row_axes[col])
            row_axes[col].set_title(label1_title)
            col += 1
        if plot_gbp:
            row_axes[col].imshow(self.xb_grad1)
            row_axes[col].set_axis_off()
            row_axes[col].set_title(label1_title)
            col += 1

        if hasattr(self, "label2"):
            label2_title = f"2.{self.label2} {self.prob2:.3f}"
            if plot_hm:
                show_heatmap(self.hmap2, self.xb_img, size, row_axes[col])
                row_axes[col].set_title(label2_title)
                col += 1
            if plot_gbp:
                row_axes[col].imshow(self.xb_grad2)
                row_axes[col].set_axis_off()
                row_axes[col].set_title(label2_title)
        # plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)


    def minmax_norm(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))


    def scaleup(x, size):
        scale_mult = size / x.shape[0]
        upsampled = scipy.ndimage.zoom(x, scale_mult)
        return upsampled


    def hooked_backward(m, xb, target_layer, clas):
        with hook_output(
            target_layer
        ) as hook_a:  
            with hook_output(
                target_layer, grad=True
            ) as hook_g:  
                preds = m(xb)
                preds[0, int(clas)].backward()  
        return hook_a, hook_g


    def clamp_gradients_hook(module, grad_in, grad_out):
        for grad in grad_in:
            torch.clamp_(grad, min=0.0)


    def hooked_ReLU(m, xb, clas):
        relu_modules = [
            module[1] for module in m.named_modules() if str(module[1]) == "ReLU(inplace)"
        ]
        with callbacks.Hooks(relu_modules, clamp_gradients_hook, is_forward=False) as _:
            preds = m(xb)
            preds[0, int(clas)].backward()


    def guided_backprop(learn, xb, y):
        xb = xb.cuda()
        m = learn.model.eval()
        xb.requires_grad_()
        if not xb.grad is None:
            xb.grad.zero_()
        hooked_ReLU(m, xb, y)
        return xb.grad[0].cpu().numpy()


    def show_heatmap(hm, xb_im, size, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        xb_im.show(ax)
        ax.imshow(
            hm, alpha=0.8, extent=(0, size, size, 0), interpolation="bilinear", cmap="magma"
        )


    def get_grad_heatmap(learn, xb, y, size):
        """
        Main function to get hmap for heatmap and xb_grad for guided backprop
        """
        xb = xb.cuda()
        m = learn.model.eval()
        target_layer = m[0][-1][-1]  
        hook_a, hook_g = hooked_backward(m, xb, target_layer, y)

        target_act = hook_a.stored[0].cpu().numpy()
        target_grad = hook_g.stored[0][0].cpu().numpy()

        mean_grad = target_grad.mean(1).mean(1)
    
        hmap = (target_act * mean_grad[..., None, None]).sum(0)
        hmap = np.where(hmap >= 0, hmap, 0)

        xb_grad = guided_backprop(learn, xb, y)  
        xb_grad = minmax_norm(xb_grad)
        hmap_scaleup = minmax_norm(scaleup(hmap, size))  


        xb_grad = np.einsum("ijk, jk->jki", xb_grad, hmap_scaleup)  

        return hmap, xb_grad


    base_dir = "DATASET"


    def get_data(sz):  
        return ImageDataBunch.from_folder(
            base_dir + "/",
            train="train",
            test="test",  
            ds_tfms=get_transforms(),
            size=sz,
            num_workers=4,
        ).normalize(
            imagenet_stats
        )  


    arch = models.resnet34
    data = get_data(224)
    learn = cnn_learner(
        data,
        arch,
        metrics=[error_rate, Precision(average="micro"), Recall(average="micro")],
        train_bn=True,
        pretrained=True,
    ).mixup()
    learn.load("ShuffleNetV2_File_4.pt")
    example_image = "../Dataset/train/Strawberry_Leaf_Scorch/image (12).JPG"

    img = open_image(example_image)

    gcam = GradCam.from_one_img(learn, img)  

    gcam.plot(
        plot_gbp=False
    )  

    gcam_heatmap = gcam.hmap1  
    from BBOXES_from_GRADCAM import BBoxerwGradCAM 

    image_resizing_scale = [400, 300]
    bbox_scaling = [1, 1, 1, 1]

    bbox = BBoxerwGradCAM(
        learn, gcam_heatmap, example_image, image_resizing_scale, bbox_scaling
    )
    for function in dir(bbox)[-18:]:
        print(function)
    bbox.show_smoothheatmap()
    bbox.show_contouredheatmap()
    bbox.show_bboxpolygon()
    bbox.show_bboxrectangle()
    rect_coords, polygon_coords = bbox.get_bboxes()
    rect_coords  
    polygon_coords


    # IoU for object detection GradCAM

    def get_IoU(truth_coords, pred_coords):
        pred_area = pred_coords[2] * pred_coords[3]
        truth_area = truth_coords[2] * truth_coords[3]
        x1 = max(truth_coords[0], pred_coords[0])
        y1 = max(truth_coords[1], pred_coords[1])
        x2 = min(truth_coords[2], pred_coords[2])
        y2 = min(truth_coords[3], pred_coords[3])
        interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        boxTruthArea = (truth_coords[2] - truth_coords[0] + 1) * (
            truth_coords[3] - truth_coords[1] + 1
        )
        boxPredArea = (pred_coords[2] - pred_coords[0] + 1) * (
            pred_coords[3] - pred_coords[1] + 1
        )
        iou = interArea / float(boxTruthArea + boxPredArea - interArea)
        return iou
    
    get_IoU([80, 40, 240, 180], rect_coords)

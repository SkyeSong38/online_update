from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from einops import rearrange
from lib.utils.image import *


class MAEACTOR(BaseActor):

	def __init__(self, net, objective, loss_weight, settings):
		super().__init__(net, objective)
		self.loss_weight = loss_weight
		self.settings = settings
		self.patchsize = self.settings.batch_size

	def __call__(self, data):
		# forward pass
		predicted_image = self.forward_pass(data)

		# process the groundtruth
		label_image = data['images']
		label = rearrange(label_image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patchsize, p2=self.patchsize)
		# compute losses
		loss, status = self.compute_losses(predicted_image, label)

		# # save reconstruction img
		# rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
		# # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
		# rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2,
		#                                                                                                             keepdim=True)
		# rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14,
		#                     w=14)
		# img = ToPILImage()(rec_img[0, :].cpu().clamp(0, 0.996))
		# img.save(f"{args.save_path}/rec_img.jpg")

		return loss, status

	def forward_pass(self, data):
		image = self.net(data['images'], data['masks'])
		return image

	def compute_losses(self, predicted_image, gt_image, return_status=True):
		# compute loss
		try:
			mse_loss = self.objective['mse'](input=predicted_image, target=gt_image)
		except:
			mse_loss = torch.tensor(0.0).cuda()

		# weighted sum
		loss = self.loss_weight['mse'] * mse_loss

		if return_status:
			# status for log
			status = {"Loss/total": loss.item(),
			          "Loss/mse": mse_loss.item()}
			return loss, status
		else:
			return loss

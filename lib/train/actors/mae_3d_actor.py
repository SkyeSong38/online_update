from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from einops import rearrange
from lib.utils.image import *


class MAE3DACTOR(BaseActor):

	def __init__(self, net, objective, loss_weight, settings):
		super().__init__(net, objective)
		self.loss_weight = loss_weight
		self.settings = settings
		self.bs = self.settings.batch_size  # batch size
		self.patchsize = self.settings.patch_size

	def __call__(self, data):
		# forward pass
		predicted_image = self.forward_pass(data)

		# process the groundtruth
		label_image = data['label_images']
		label_image = rearrange(label_image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patchsize,
		                        p2=self.patchsize)
		gt_image = data['gt_images']
		gt_image = rearrange(gt_image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patchsize, p2=self.patchsize)
		# compute losses
		loss, status = self.compute_losses(predicted_image, gt_image, label_image)

		# # save reconstruction img
		# rec_img = rearrange(predicted_image, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=3, p1=self.patchsize,
		#                     p2=self.patchsize, h=14, w=14)
		# draw_seq_image(data['seq_images'][0], rec_img[0], norm_img=True)
		return loss, status

	def forward_pass(self, data):
		image = self.net(data['seq_images'], data['seq_att'])
		return image

	def compute_losses(self, predicted_image, gt_image, label_image, return_status=True):
		# compute loss
		try:
			mse_loss1 = self.objective['mse'](input=predicted_image, target=label_image)
			mse_loss2 = self.objective['mse'](input=predicted_image, target=gt_image)
			mse_loss = 0.5 * mse_loss1 + 0.5 * mse_loss2
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

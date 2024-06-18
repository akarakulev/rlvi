"""
ResNet: resnet50, pre-trained on ImageNet
"""

import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, outplanes, stride=1, downsample=None, dropout_rate=0):
		super().__init__()

		self.dropout_rate = dropout_rate

		self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(outplanes)
		
		self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(outplanes)

		self.conv3 = nn.Conv2d(outplanes, outplanes*self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(outplanes*self.expansion)

		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x): 
		residual = x 
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		if self.dropout_rate > 0:
			out = nn.functional.dropout2d(out, p=self.dropout_rate)
		out = self.conv3(out)
		out = self.bn3(out)
		if(self.downsample is not None):
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)

		return out



class ResNet(nn.Module):

	def __init__(self, block, layer, num_classes, input_channel, dropout_rate=0):
		super().__init__()
		self.inplanes = 64
		self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias = False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer1 = self.make_layer(block, 64, layer[0], dropout_rate=dropout_rate)
		self.layer2 = self.make_layer(block, 128, layer[1], stride=2, dropout_rate=dropout_rate)
		self.layer3 = self.make_layer(block, 256, layer[2], stride=2, dropout_rate=dropout_rate)
		self.layer4 = self.make_layer(block, 512, layer[3], stride=2, dropout_rate=dropout_rate)

		self.avgpool = nn.AvgPool2d(7, stride = 2)

		self.dropout = nn.Dropout2d(p = 0.5, inplace = True)

		self.fc = nn.Linear(512*block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2./n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def make_layer(self, block, planes, blocks, stride=1, dropout_rate=0):
		downsample = None

		if(stride !=1 or self.inplanes != planes * block.expansion):
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), 
				nn.BatchNorm2d(planes*block.expansion))

		layers = []

		layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=dropout_rate))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))


		return nn.Sequential(*layers)

	def fc_params(self):
		params = []
		for name, param in self.named_parameters():
			if 'fc' in name:
				params.append(param)
		return params

	def backbone_params(self):
		params = []
		for name, param in self.named_parameters():
			if 'fc' not in name:
				params.append(param)
		return params

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = self.dropout(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x


def resnet50(num_classes, input_channel, pretrained=True, dropout_rate=0):
	model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_channel, dropout_rate=dropout_rate)

	if pretrained == True:
		state_dict = model_zoo.load_url(
			'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h2_176-001a1197.pth', 
			map_location='cpu'
		)
		state_dict.pop('fc.weight', None)
		state_dict.pop('fc.bias', None)
		model.load_state_dict(state_dict, strict=False)
		print("Loaded Imagenet pretrained model")

	return model

import torch
from torchvision.models import resnet50, ResNet18_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
import torch.nn as nn
#
# # To assist you in designing the feature extractor you may want to print out
# # the available nodes for resnet50.
# # m = resnet50()
# # train_nodes, eval_nodes = get_graph_node_names(resnet50())
#
# # The lists returned, are the names of all the graph nodes (in order of
# # execution) for the input model traced in train mode and in eval mode
# # respectively. You'll find that `train_nodes` and `eval_nodes` are the same
# # for this example. But if the model contains control flow that's dependent
# # on the training mode, they may be different.
#
# # To specify the nodes you want to extract, you could select the final node
# # that appears in each of the main layers:
# return_nodes = {
#     # node_name: user-specified key for output dict
#     'layer1.2.relu_2': 'layer1',
#     'layer2.3.relu_2': 'layer2',
#     'layer3.5.relu_2': 'layer3',
#     'layer4.2.relu_2': 'layer4',
# }
#
# # But `create_feature_extractor` can also accept truncated node specifications
# # like "layer1", as it will just pick the last node that's a descendent of
# # of the specification. (Tip: be careful with this, especially when a layer
# # has multiple outputs. It's not always guaranteed that the last operation
# # performed is the one that corresponds to the output you desire. You should
# # consult the source code for the input model to confirm.)
# return_nodes = {
#     'layer1': 'layer1',
#     'layer2': 'layer2',
#     'layer3': 'layer3',
#     'layer4': 'layer4',
# }
#
# # Now you can build the feature extractor. This returns a module whose forward
# # method returns a dictionary like:
# # {
# #     'layer1': output of layer 1,
# #     'layer2': output of layer 2,
# #     'layer3': output of layer 3,
# #     'layer4': output of layer 4,
# # }
# create_feature_extractor(m, return_nodes=return_nodes)
#
# # Let's put all that together to wrap resnet50 with MaskRCNN
#
# # MaskRCNN requires a backbone with an attached FPN
# class Resnet50WithFPN(torch.nn.Module):
#     def __init__(self):
#         super(Resnet50WithFPN, self).__init__()
#         # Get a resnet50 backbone
#         m = resnet50()
#         # Extract 4 main layers (note: MaskRCNN needs this particular name
#         # mapping for return nodes)
#         self.body = create_feature_extractor(
#             m, return_nodes={f'layer{k}': str(v)
#                              for v, k in enumerate([1, 2, 3, 4])})
#         # Dry run to get number of channels for FPN
#         inp = torch.randn(2, 3, 224, 224)
#         with torch.no_grad():
#             out = self.body(inp)
#         in_channels_list = [o.shape[1] for o in out.values()]
#         # Build FPN
#         self.out_channels = 256
#         self.fpn = FeaturePyramidNetwork(
#             in_channels_list, out_channels=self.out_channels,
#             extra_blocks=LastLevelMaxPool())
#
#     def forward(self, x):
#         x = self.body(x)
#         x = self.fpn(x)
#         return x


# Now we can build our model!
# model = MaskRCNN(Resnet50WithFPN(), num_classes=91).eval()

from torch.autograd import Variable
from torchvision import models
# rn18 = models.resnet18(pretrained=True)
# children_counter = 0
# for n,c in rn18.named_children():
#     print("Children Counter: ",children_counter," Layer Name: ",n,)
#     children_counter+=1
#
# print(rn18._modules)

class Truncate_Resnet(nn.Module):
    def __init__(self, output_layer=None, in_channels = 3, out_channels = 3):
        super().__init__()
        self.pretrained = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers) - self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        self.net = nn.Sequential(self.pretrained._modules)
        for param in self.net.parameters():
            param.requires_grad = False
        self.pretrained = None

    def forward(self, x):
        x = self.net(x)
        return x

# model4 = Truncate_Resnet(output_layer ='layer4')
# model3 = Truncate_Resnet(output_layer ='layer3')
# model2 = Truncate_Resnet(output_layer ='layer2')
# model1 = Truncate_Resnet(output_layer ='layer1')
#
#
# from torchsummary import summary
# summary(model1,input_size=(3, 224, 224))
# x = torch.randn((3, 3, 160, 160))
# skip_connections = []
# x1 = model1(x)
# skip_connections.append(x1)
# x2 = model2(x)
# skip_connections.append(x2)
#
# x3 = model3(x)
# skip_connections.append(x3)
#
# x4 = model4(x)
# skip_connections.append(x4)
# #
# print(skip_connections)
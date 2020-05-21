import torch
import model
import collections

old_dict = torch.load('./checkpoint/vgg16.pth')
new_dict = collections.OrderedDict()
VGG = model.VGGBackbone()
new_dict["layer1.0.weight"] = old_dict["features.0.weight"]
new_dict["layer1.0.bias"] = old_dict["features.0.bias"]
new_dict["layer1.2.weight"] = old_dict["features.2.weight"]
new_dict["layer1.2.bias"] = old_dict["features.2.bias"]

new_dict["layer2.0.weight"] = old_dict["features.5.weight"]
new_dict["layer2.0.bias"] = old_dict["features.5.bias"]
new_dict["layer2.2.weight"] = old_dict["features.7.weight"]
new_dict["layer2.2.bias"] = old_dict["features.7.bias"]

new_dict["layer3.0.weight"] = old_dict["features.10.weight"]
new_dict["layer3.0.bias"] = old_dict["features.10.bias"]
new_dict["layer3.2.weight"] = old_dict["features.12.weight"]
new_dict["layer3.2.bias"] = old_dict["features.12.bias"]
new_dict["layer3.4.weight"] = old_dict["features.14.weight"]
new_dict["layer3.4.bias"] = old_dict["features.14.bias"]

new_dict["layer4.0.weight"] = old_dict["features.17.weight"]
new_dict["layer4.0.bias"] = old_dict["features.17.bias"]
new_dict["layer4.2.weight"] = old_dict["features.19.weight"]
new_dict["layer4.2.bias"] = old_dict["features.19.bias"]
new_dict["layer4.4.weight"] = old_dict["features.21.weight"]
new_dict["layer4.4.bias"] = old_dict["features.21.bias"]

new_dict["layer5.0.weight"] = old_dict["features.24.weight"]
new_dict["layer5.0.bias"] = old_dict["features.24.bias"]
new_dict["layer5.2.weight"] = old_dict["features.26.weight"]
new_dict["layer5.2.bias"] = old_dict["features.26.bias"]
new_dict["layer5.4.weight"] = old_dict["features.28.weight"]
new_dict["layer5.4.bias"] = old_dict["features.28.bias"]
VGG.load_state_dict(new_dict)
torch.save(VGG.state_dict(), './checkpoint/vggbb.pth')
# print(type(old_dict))
# for k, v in old_dict.items():
#     print(k)

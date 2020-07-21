import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from training_dense_net import char_list


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = float(drop_rate)

    def forward(self, inputs):
        # inputs are a list of [last output, last input]
        out = torch.cat(inputs, 1)
        out = F.relu(self.norm1(out))
        out = self.conv1(out)
        out = F.relu(self.norm2(out))
        out = self.conv2(out)
        return out


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, input):
        out = [input]
        for name, layer in self.items():
            new_out = layer(out)
            out.append(new_out)
        return torch.cat(out, 1)


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features, pool_size=(2, 1)):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.add_module('relu', nn.ReLU(inplace=True))
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, input):
        out = F.relu(self.norm(input))
        out = self.conv(out)
        out = self.pool(out)
        return out


class _EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(_EncoderRNN, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)

    def forward(self, input):
        return self.lstm(input)


class _DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(_DecoderRNN, self).__init__()

        self.lstm = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True)
        self.out = nn.Linear(2 * hidden_size, output_size)

    def forward(self, input):
        output = self.lstm(*input)
        output = self.out(output[0])
        return output


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, bn_size=4):
        super(DenseNet, self).__init__()

        num_features = 2 * growth_rate
        self.conv1 = nn.Conv2d(1, num_features, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.BatchNorm2d(num_features)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        num_layers = 6
        self.dense1 = _DenseBlock(num_layers, num_features, bn_size, growth_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition1 = _Transition(num_features, num_features // 2, (2, 2))
        num_features = num_features // 2

        num_layers = 12
        self.dense2 = _DenseBlock(num_layers, num_features, bn_size, growth_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition2 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        num_layers = 48
        self.dense3 = _DenseBlock(num_layers, num_features, bn_size, growth_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition3 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        num_layers = 32
        self.dense4 = _DenseBlock(num_layers, num_features, bn_size, growth_rate)
        num_features = num_features + num_layers * growth_rate
        self.final_features = num_features
        self.norm5 = nn.BatchNorm2d(num_features)

        # self.linear = nn.Linear(num_features, len(char_list) + 1)
        self.encoder = _EncoderRNN(num_features, 128)
        self.decoder = _DecoderRNN(128, len(char_list) + 1)
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, input):
        out = F.relu(self.conv1(input))
        # print(f"conv: {out.shape}")
        out = self.norm1(out)
        out = self.pool1(out)
        # print(f"shape 1: {out.shape}")
        out = self.dense1(out)
        out = self.transition1(out)
        # print(f"shape 2: {out.shape}")
        out = self.dense2(out)
        out = self.transition2(out)
        # print(f"shape 3: {out.shape}")
        out = self.dense3(out)
        out = self.transition3(out)
        # print(f"shape 4: {out.shape}")
        out = self.dense4(out)
        # print(f"shape 5: {out.shape}")
        out = F.relu(self.norm5(out))
        # out = self.pool5(out)
        # out = out.view((-1, self.final_features))
        out = torch.squeeze(out, 2).permute(0, 2, 1)
        # print(f"shape 6: {out.shape}")
        # out = self.linear(out)
        out = self.encoder(out)
        # print(f"encoder shape: {out[0].shape}")
        out = self.decoder(out)
        # print(f"decoder shape: {out.shape}")
        # print(f"shape 7: {out.shape}")
        out = self.log_softmax(out)
        return out

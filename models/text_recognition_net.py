###################################################################################################
# Deep Vision Project: Text Extraction from Receipts
#
# Authors: Benjamin Maier and Hauke Lüdemann
# Data: 2020
#
# Description of file:
#   Implementation of the text recognition models.
###################################################################################################



import torch.nn.functional as F
import torch.nn as nn
import torch

char_list = "abcdefghjiklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.:,/\*!&?%()-_ "
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    """ Dense Net for text recognition.
    """

    def __init__(self, type="LinearAttention", out_put_size=64, growth_rate=32, bn_size=4):
        """ Constructor for Dense Net.

        :param type: Type specifying the final layer of the Dense net. One of "LinearAttention" or "RNNAttention".
        :param out_put_size: Maximal word length that can be recognized. This is also the input size for the CTC loss.
                             Currently 32 and 64 are supported.
        :param growth_rate: ...
        :param bn_size: ...
        """
        super(DenseNet, self).__init__()

        assert out_put_size == 64 or out_put_size == 32, "Unsupported output size!"
        assert type == "LinearAttention" or type == "RNNAttention", "Unsupported attention layer specified in type!"

        self.type = type

        num_features = 2 * growth_rate
        self.conv1 = nn.Conv2d(1, num_features, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.BatchNorm2d(num_features)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        num_layers = 6
        self.dense1 = _DenseBlock(num_layers, num_features, bn_size, growth_rate)
        num_features = num_features + num_layers * growth_rate
        if out_put_size == 32:
            self.transition1 = _Transition(num_features, num_features // 2, (2, 2))
        elif out_put_size == 64:
            self.transition1 = _Transition(num_features, num_features // 2)
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

        if self.type == "LinearAttention":
            self.linear = nn.Linear(num_features, len(char_list) + 1)
        elif self.type == "RNNAttention":
            self.encoder = _EncoderRNN(num_features, 128)
            self.decoder = _DecoderRNN(128, len(char_list) + 1)

        self.log_softmax = nn.LogSoftmax(2)

    def forward(self, input):
        out = F.relu(self.conv1(input))
        out = self.norm1(out)
        out = self.pool1(out)

        out = self.dense1(out)
        out = self.transition1(out)

        out = self.dense2(out)
        out = self.transition2(out)

        out = self.dense3(out)
        out = self.transition3(out)

        out = self.dense4(out)
        out = F.relu(self.norm5(out))

        out = torch.squeeze(out, 2).permute(0, 2, 1)

        if self.type == "LinearAttention":
            out = self.linear(out)
        elif self.type == "RNNAttention":
            out = self.encoder(out)
            out = self.decoder(out)

        out = self.log_softmax(out)

        return out


class CRNN(nn.Module):
    """ CRNN Net for text recognition.
    """

    def __init__(self, out_put_size=64):
        """ Constructor of CRNN

        :param out_put_size: Maximal word length that can be recognized. This is also the input size for the CTC loss.
                             Currently 32 and 64 are supported.
        """
        super(CRNN, self).__init__()

        assert out_put_size == 64 or out_put_size == 32, "Unsupported output size!"

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)

        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)

        if out_put_size == 32:
            self.pool4 = nn.MaxPool2d(2, stride=2)
        else:
            self.pool4 = nn.MaxPool2d((2, 1), stride=(2, 1))

        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d((2, 1), stride=(2, 1))

        self.conv7 = nn.Conv2d(512, 512, 2)

        self.encoder = _EncoderRNN(512, 128)
        self.decoder = _DecoderRNN(128, len(char_list) + 1)
        self.log_softmax = nn.LogSoftmax(2)

    def forward(self, input):

        out = self.pool1(F.relu(self.conv1(input)))
        out = self.pool2(F.relu(self.conv2(out)))

        out = F.relu(self.conv3(out))

        out = self.pool4(F.relu(self.conv4(out)))
        out = self.bn5(F.relu(self.conv5(out)))
        out = self.pool6(self.bn6(F.relu(self.conv6(out))))

        out = F.relu(self.conv7(out))

        out = torch.squeeze(out, 2).permute(0, 2, 1)

        out = self.encoder(out)
        out = self.decoder(out)

        out = self.log_softmax(out)

        return out


def load_text_recognition_model(model_name, model_weights=None, out_put_size=64):
    """ Load specified text recognition model.

    :param model_name: Name of the model. One of "DenseNetLinear", "DenseNetRNN", "CRNN"
    :param model_weights: Checkpoint of model.
    :param out_put_size: Maximal word length that can be recognized. This is also the input size for the CTC loss.
                             Currently 32 and 64 are supported.
    :return: Model.
    """

    assert model_name == "DenseNetLinear" or model_name == "DenseNetRNN" or model_name == "CRNN", \
        "Unsupported model name specified!"

    if model_name == "DenseNetLinear":
        model = DenseNet(type="LinearAttention", out_put_size=out_put_size)
    elif model_name == "DenseNetRNN":
        model = DenseNet(type="RNNAttention", out_put_size=out_put_size)
    else:
        model = CRNN(out_put_size=out_put_size)

    if model_weights is not None:
        model.load_state_dict(torch.load(model_weights, map_location=device))

    return model

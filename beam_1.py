import torch
import torch.nn.functional as f
import math


def max_path(candidate):
    v = -999
    p = 0 # position of max
    for i in range(len(candidate)):
        if candidate[i][-1] > v:
            v = candidate[i][-1]
            p = i
    return p


def beam_search(x, model, config):
    h, encoder_out = model.encoder(x)
    # init
    path = [[[config.bos], h, 0.0]]
    for i in range(config.s_len):
        candidate = []
        for j in range(len(path)):
            out = torch.tensor(path[j][0][-1]).type(torch.LongTensor).unsqueeze(0)
            h = path[j][1]
            # print(h[0].size())
            _, out, h = model.decoder(out, h, encoder_out)
            data = f.softmax(model.output_layer(out), -1)
            sorted, indices = torch.sort(data, descending=True)
            sorted = sorted.squeeze()
            indices = indices.squeeze()
            pre_path = path[j][0]
            pre_prob = path[j][-1]
            for k in range(config.beam_size):
                p = pre_path.copy()
                p.append(int(indices[k].item()))
                prob = math.log(sorted[k]) + pre_prob
                candidate.append([p, h, prob])
        path = []
        if i == config.s_len - 1:
            r = candidate.pop(max_path(candidate))
            return r[0]
        for z in range(config.beam_size):
            path.append(candidate.pop(max_path(candidate)))

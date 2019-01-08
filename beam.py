import torch
import math


def maximum(data):
    # position
    p = torch.argmax(data).item()
    # value
    v = data[p].item()
    data[p] = torch.tensor(0.0)
    return p, v, data


def max_prob(candidate):
    v = 0
    p = 0 # position of max
    for i in range(len(candidate)):
        if candidate[i][1] > v:
            v = candidate[i][1]
            p = i
    return p


# beam search
def beam_search(datas, start, beam_size, s_len):
    # Initiliazation
    sequence = [[[start], 0.0]]
    # sequence length
    for i in range(s_len):
        # find the next node
        candidate = []
        for j in range(len(sequence)):
            # find the maximum(beam size)
            data = datas[i]      # data = sequence[j][0][-1] -> tensor
            pre_path = sequence[j][0]
            pre_prob = sequence[j][1]
            for k in range(beam_size):
                p, v, data = maximum(data)
                path = pre_path.copy()
                path.append(p)
                prob = math.log(v) + pre_prob
                candidate.append([path, prob])
        sequence = []
        for w in range(beam_size):
            sequence.append(candidate.pop(max_prob(candidate)))
    return sequence




import numpy as np


def decode_topk_tx(
    out_tx, out_2d_idxs, out_1d_len, out_2d_len, k=1, pass_score=0
):
    nodes = np.arange(out_2d_len)
    out_tx[nodes[:-1], nodes[1:]] = np.maximum(
        pass_score, out_tx[nodes[:-1], nodes[1:]]
    )

    dp_score = np.full((out_2d_len, k), -np.inf, dtype=np.float32)
    dp_score[0, -1] = 0
    dp_node = np.full((out_2d_len, k), -1, dtype=np.int32)
    dp_rank = np.full((out_2d_len, k), -1, dtype=np.int32)
    topk_tx = np.ones((k, out_1d_len), dtype=np.int32)

    for node in range(1, out_2d_len):
        score = (dp_score[:node] + out_tx[:node, node:node + 1]).reshape(-1)
        node_rank = np.argsort(score)[-k:]
        dp_score[node] = score[node_rank]
        dp_node[node] = node_rank // k
        dp_rank[node] = node_rank % k

    for k_idx in range(k):
        node, rank = out_2d_len - 1, k - 1 - k_idx
        while node > 0:
            if out_tx[dp_node[node, rank], node] != pass_score:
                topk_tx[
                    k_idx,
                    out_2d_idxs[dp_node[node, rank]]:out_2d_idxs[node] + 1
                ] = 0
            node, rank = dp_node[node, rank], dp_rank[node, rank]

    return topk_tx

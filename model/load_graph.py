import dgl
import numpy as np

def load_graph(user_num) -> list:
    records = []
    f = open("data/train.txt", "r")
    for line in f:
        record = line.strip().split("\t")
        records.append(record)

    init_item_init = []
    init_item_item = []
    init_part_init = []
    init_part_part = []
    part_item_part = []
    part_item_item = []

    for record in records:
        init_item_init.append(int(record[0]))
        init_item_item.append(int(record[1]) + user_num)
        for p in record[2:]:
            init_part_init.append(int(record[0]))
            init_part_part.append(int(p))
            part_item_part.append(int(p))
            part_item_item.append(int(record[1]) + user_num)

    records = []
    f = open("data/valid.txt", "r")
    for line in f:
        record = line.strip().split("\t")
        records.append(record)

    for record in records:
        init_item_init.append(int(record[0]))
        init_item_item.append(int(record[1]) + user_num)

    init_item_graph = dgl.DGLGraph((np.array(init_item_init), np.array(init_item_item)))
    init_part_graph = dgl.DGLGraph((np.array(init_part_init), np.array(init_part_part)))
    part_item_graph = dgl.DGLGraph((np.array(part_item_part), np.array(part_item_item)))

    g_edges = init_item_graph.edges()
    init_item_graph.add_edges(g_edges[1], g_edges[0])
    g_edges = init_part_graph.edges()
    init_part_graph.add_edges(g_edges[1], g_edges[0])
    g_edges = part_item_graph.edges()
    part_item_graph.add_edges(g_edges[1], g_edges[0])

    print(init_item_graph.number_of_nodes())
    print(init_part_graph.number_of_nodes())
    print(part_item_graph.number_of_nodes())

    return [init_item_graph, init_part_graph, part_item_graph]

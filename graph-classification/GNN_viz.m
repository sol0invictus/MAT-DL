function GNN_viz(idx,Adj)
    G = graph(Adj{idx});
    plot(G,'Layout','force')
end
%Custom loader for Proteins data-set
function [final_adj,final_feat,final_label,adj_unpro]=protein_load()
    graph_num = load('dataset/PROTEINS_graph_indicator.txt');
    graph_nodes = load('dataset/PROTEINS_A.txt');
    graph_labels = load('dataset/PROTEINS_graph_labels.txt');
    node_att = load('dataset/PROTEINS_node_attributes.txt');
    max(node_att);
    node_att = normalize(node_att,'scale');
    %Now we will load the starting and end nodenumbers of each graph
    temp = 1;
    node_range(1,1) = 1;

    for i=1:length(graph_num)
        if graph_num(i) ~= temp
            node_range(temp,2)= i-1;
            node_range(temp+1,1) = i;
            temp = temp + 1;
        end
    end

    node_range(temp,2) = length(graph_num);
    
    % Create adjaceny Matrix
    adj=cell(length(node_range),1);
    feat=cell(length(node_range),1);
    graph_num = 1;
    adj{1} = zeros(node_range(1,2)-node_range(1,1));
    for i=1:length(graph_nodes)
        %adj{graph_num}=zeros(node_range(graph_num,2)-node_range(graph_num,1));
        if graph_nodes(i,1) <= node_range(graph_num,2)
            offset = 0;
            if graph_num>1
                offset = node_range(graph_num-1,2);
            end
            x = graph_nodes(i,1) - offset;
            y = graph_nodes(i,2) - offset;
            adj{graph_num}(x,y) = 1;
            feat{graph_num}(x)=node_att(graph_nodes(i,1));
        else 
            graph_num = graph_num + 1;
            adj{graph_num} = zeros(node_range(graph_num,2)-node_range(graph_num,1));
            offset = 0;
            if graph_num>1
                offset = node_range(graph_num-1,2);
            end
            x = graph_nodes(i,1) - offset;
            y = graph_nodes(i,2) - offset;
            adj{graph_num}(x,y) = 1;
        end
    end
    
    %Consider only graphs with nodes <=50
    % These parameters dictate the graph size range
    lower = 1;
    higher = 50;
    
    counter = 1;
    final_adj={};
    final_label=[];
    final_feat={};
    adj_unpro={};
    for i=1:length(node_range)
       if length(adj{i}) >= lower && length(adj{i}) <= higher
        final_label(counter)= graph_labels(i);
        final_adj{counter} = zeros([higher,higher]);
        final_feat{counter} = zeros([higher,1]);
        temp = length(adj{i});
        %adjacency matrix pre-processing
        ahat = adj{i} + eye(temp);
        dg = diag(sum(ahat));
        ahat2 = dg^(-1/2) * ahat * dg^(-1/2);
        adj_unpro{counter} = adj{i};
        final_adj{counter}(1:temp,1:temp) = ahat2;
        final_feat{counter}(1:temp,1) = feat{i};
        counter = counter + 1;
       end    
    end
end

classdef GCN < nnet.layer.Layer
	properties (Learnable)
		graph_weight;
        graph_bias;
	end
	
	methods
		function layer = GCN(layer_size,name)
			layer.Name = name;
			layer.Description = "GCN Layer";
			layer.graph_weight = dlarray(rand(layer_size)*2-1);
            layer.graph_bias = dlarray(zeros([1,layer_size(2)]));
            layer.NumInputs = 2;
            layer.NumOutputs = 2;
		end
		
		function [A,Z] = predict(layer,A,H)
			W = layer.graph_weight;
			Z = sigmoid(A*H*W);
		end
		
		function [A,Z] = forward(layer,A,H)
			W = layer.graph_weight;
			Z = sigmoid(A*H*W);
		end
	end
end
classdef GNN_interface < nnet.layer.Layer
	properties
        nodes;
    end
	methods
		function layer = GNN_interface(size,name)
			layer.Name = name;
			layer.Description = "Reorders for subsequent FC layers";
            layer.NumOutputs = 1;
            layer.NumInputs = 2;
            layer.nodes = size;
		end
		
		function Z = predict(layer,A,H)
            Z = reshape(H,[layer.nodes,1]);
        end
		
		function Z = forward(layer,A,H)
			Z = reshape(H,[layer.nodes,1]);
        end
	end
end
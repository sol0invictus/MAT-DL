function [gradients,loss] = GNNGradients(dlnet,adj,feat,target)

out = forward(dlnet,adj,feat);
loss = crossentropy(out,target,'TargetCategories','independent');
gradients = dlgradient(loss,dlnet.Learnables);

end


layers = [
    featureInputLayer(2,"Name","input_1")
    fullyConnectedLayer(24,"Name","fc_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(48,"Name","fc_2")
    reluLayer("Name","relu_2")
    fullyConnectedLayer(3,"Name","output")
    regressionLayer("Name","RepresentationLoss")];
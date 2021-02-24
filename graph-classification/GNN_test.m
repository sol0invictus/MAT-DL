function acc = GNN_test(dlnet,adj,feat,label,test_idx)
acc = 0;
pred=[];
test_len = length(test_idx)
    for i=1:test_len
    var1=dlarray(adj{test_idx(i)},'CU');
    var2=dlarray(feat{test_idx(i)},'C');
    var3=dlarray(label(test_idx(i))-1,'CB');
    out=round(predict(dlnet,var1,var2));
    if out==var3
        acc = acc +1;
    end
    pred(i)=out;
    end  
    acc =acc*100 /length(test_idx);
    C=confusionmat(pred',(label(test_idx)-1)');
    confusionchart(C)
end
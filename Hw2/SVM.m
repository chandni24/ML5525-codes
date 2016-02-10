function [pred] = SVM(i)

    global X Y K L_alpha;
    pred = 0;
    
    %compute the predicted value of the class for i th training sample
    pred = sum(Y .* L_alpha .* K(i,:)');

    
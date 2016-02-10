function [objFncV] = objFnc_SMO

    global X Y K L_alpha;
    M = length(X);

    %objFncV = sum(L_alpha) - L_alpha'*diag(Y)*K*diag(Y)*L_alpha;
    
    term1 = sum(L_alpha);

    for i = 1:M
        for j = 1:M
            term2 = sum(Y(i) * Y(j) * K(i, j) * L_alpha(i) * L_alpha(j)) / 2;
        end
    end

    %compute the objective function at this point (i.e. after the alpha value for the two chosen points is changed)
    objFncV = (term1 - term2);

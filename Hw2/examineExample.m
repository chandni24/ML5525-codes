function [a] = examineExample(i2)

global X Y E eps C L_alpha;
   %initializing variables
y2 = Y(i2);
alph2 = L_alpha(i2);
E2 = E(i2);
r2 = E2*y2;
tol = 10^(-7);

a = 0;%denotes update is done

%run only for samples where the KKT conditions are being violated
if( r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0)
    % find non zero indexes
    nonZeroIndx = find (L_alpha ~= 0 & (L_alpha < C | L_alpha > -C));
    if ~isempty(nonZeroIndx)

        % find second choice heuristic to maximize step size
        % choose max of negative error if label positive, and vice versa
        % sort in ascending order, and get indexes of sort order
        [~, index] = sort(E);
        
        %if i2 is minimum, select 2nd minimum
        if E(i2) > 0
            if index(1) == i2
                i1 = index(2);
            else
                i1 = index(1);
            end
        %if i2 is maximum, select 2nd last maximum
        else
            if (index(end) == i2)
                i1 = index(end-1);
            else
                i1 = index(end);
            end
        end
        
        % if update is done, successful heuristic approximation
        if (takeStep(i1,i2) == 1)
            a = 1;
            return;
        end

        % loop over all possible non zero and non-c alpha, starting at random point
        randIndx = randperm(length(nonZeroIndx));
        for j=1:length(nonZeroIndx)
            if (takeStep(randIndx(j),i2) == 1)
                a = 1;
                return
            end
        end
    end

    % loop over all possible i1, starting at random point
    randIndx = randperm(length(L_alpha));
    for j=1:length(L_alpha)
        if (takeStep(randIndx(j),i2) == 1)
            a = 1;
            return;
        end
    end  
end

return;

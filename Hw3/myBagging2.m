
function [output output2] = myBagging2(filename, B)

    %clearvars;

    %input_matrix = csvread('D:\MachineLearning5525\HW3\Mushroom.csv');

    input_matrix = csvread(filename);

    num_crossval = 10;

    [M, N] = size(input_matrix);
    %B = [5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100];% 105 110 115 120 125 130 135 140 145 150];

    %global no_of_features un featureSeq y_pred;
    no_of_features = size(input_matrix,2)-1;

    un = cell(1,size(input_matrix,2));
    for x = 1:size(input_matrix,2)
        un(x) = {unique(input_matrix(:,x))};
    end

    test_SampleSize = floor(M/num_crossval);

    input_data = input_matrix;

    for x = 1:size(B,2)
        x
        noOfBases = B(x);
        %x=1;
        for noOfSampleGroup = 1:num_crossval

            for noOfBaseClassifiers = 1:noOfBases

                clear Entropy_dash;

                %input_matrix = input_data(randperm(size(input_data,1)),:);

                X_test = input_matrix((noOfSampleGroup-1)*test_SampleSize+1:noOfSampleGroup*test_SampleSize, :);

                if noOfSampleGroup == 1
                    %xTraining = x(((index*sizeMatrix)+1):m,:);
                    X_train = input_matrix((noOfSampleGroup*test_SampleSize+1):M, :);
                else
                    X_train = input_matrix(1:(noOfSampleGroup-1)*test_SampleSize, :);
                    X_train = [X_train;input_matrix(noOfSampleGroup*test_SampleSize+1:M, :)];
                end

                %X = X_train;
                training_SampleSize = M-test_SampleSize;
                %X_train = datasample(X_train, training_SampleSize);

                y_train = zeros(training_SampleSize,noOfBaseClassifiers);

                noOfLayers = 2;

                [X, index] = datasample(X_train, training_SampleSize);

                Entropy_parent = -(((sum(X(:,1)==1)/training_SampleSize) * log2(sum(X(:,1)==1)/training_SampleSize)) + ((sum(X(:,1)==-1)/training_SampleSize) * log2(sum(X(:,1)==-1)/training_SampleSize)));
                Entropy_dash{1} = Entropy_parent;

                for n = 1:noOfLayers

                    if n==1
                        sizeOfLayer = 1;
                    elseif n~=1
                        sizeOfLayer = size(X_dash, 2);
                    end

                    for node = 1:sizeOfLayer

                        if n~=1 && Entropy_dash(node) == 0
                            x_pred = X(X_dash{node}, 1);
                            %pred{node} = (sum(x_pred==1)> sum(x_pred==-1));
                            %                         if sum(x_pred==1)> sum(x_pred==-1)
                            %                             pred{featureSeq{1,1}}{node} = 1 %at a layer
                            %                         else
                            %                             pred{featureSeq{1,1}}{node} = -1
                            %                         end
                            continue;
                        end

                        if n==1
                            X_curr = X;
                        elseif n~=1
                            X_curr = X(X_dash{node}, :);
                            Entropy_parent = Entropy_dash(node);
                            pred{1}{node}={};
                        end

                        maxGain = 0;

                        for i = 2:no_of_features+1 %calculate the entropy for each node and find maximum information gain
                            A = zeros(size(un{i},1), 3);

                            Entropy = zeros(size(un{i},1),1);
                            %A would have vectors for the nodes, with
                            %index 0 - positive class samples
                            %index 1 - negative class samples
                            %index 2 - entropy

                            total_entropy = 0;

                            for j = 1:size(un{i},1) %for first feature j takes 6 values
                                A(j,2) = sum(X_curr(find(X_curr(:,i)==j))==1);
                                A(j,3) = sum(X_curr(find(X_curr(:,i)==j))==-1);
                                A(j,1) = A(j,2) + A(j,3);
                                %Entropy(j) = -( + ((A(j,3)/training_SampleSize) * log2(A(j,3)/training_SampleSize)));
                                if ~(isnan((A(j,2)/A(j,1)) * log2(A(j,2)/A(j,1))))
                                    Entropy(j) = Entropy(j) - ((A(j,2)/A(j,1)) * log2(A(j,2)/A(j,1)));
                                end
                                if ~(isnan((A(j,3)/A(j,1)) * log2(A(j,3)/A(j,1))))
                                    Entropy(j) = Entropy(j) - ((A(j,3)/A(j,1)) * log2(A(j,3)/A(j,1)));
                                end
                                total_entropy = total_entropy + (A(j,1)/training_SampleSize)*Entropy(j);
                            end

                            Entropy_parent - total_entropy;

                            if (maxGain < (Entropy_parent - total_entropy))
                                maxGain = Entropy_parent - total_entropy;
                                clear X_next_dash pred_next;
                                featureSeq{n, node} = i;
                                for j = 1:size(un{i},1)
                                    X_next_dash{j} = find(X_curr(:,i)==j);
                                    if sum(X_curr(X_next_dash{j},1)==1)> sum(X_curr(X_next_dash{j},1)==-1)
                                        pred_next{j} = 1; %at a layer
                                    else
                                        pred_next{j} = -1;
                                    end
                                end
                                if n~=1
                                    x_pred = X(X_dash{node}, 1);
                                    %pred_next(node) = max(sum(x_pred==1), sum(x_pred==-1));
                                end
                                Entropy_next_dash = Entropy;
                            end

                        end  %noOfFeatures

                        if exist('pred_next') && n==1%&& cellfun(@isempty,pred_next) = ones
                            pred{node} = pred_next;
                        else
                            pred{1}{node} = pred_next;
                        end
                    end  %nodes in a layer

                    if exist('X_next_dash')
                        X_dash = X_next_dash;
                    end
                    if exist('Entropy_next_dash')
                        Entropy_dash = Entropy_next_dash;
                    end
                    %pred() =

                end  %layers

                %works only for two layers
                %y_pred = zeros(size(un{featureSeq(1)}, 1), size(un{featureSeq(2)}, 1));
                for i = 1:size(un{featureSeq{1,1}},1)%size(pred,2)
                    for j = 1:size(un{featureSeq{2,featureSeq{1,1}}},1)%size(pred{i},2)
                        if iscell(pred{1}{i}) ~= 1
                            y_pred(i,j) = pred{1}{i};
                        else
                            y_pred(i,j) = pred{1}{i}{j};
                        end
                    end
                end

                featureSeq{2,max(cell2mat(cellfun(@size,un,'uni',false)))} = [];

                for i = 1:size(X,1)
                    x_cord = X(i,featureSeq{1,1});
                    if isempty(featureSeq{2,x_cord})
                        y_cord = 1;
                    else
                        y_cord = X(i, featureSeq{2,x_cord});
                    end
                    if iscell(pred{1}{x_cord}) ~= 1
                        y_train(index(i),noOfBaseClassifiers) = pred{1}{x_cord};%y_pred(x_cord, y_cord);
                    else
                        y_train(index(i),noOfBaseClassifiers) = pred{1}{x_cord}{y_cord};
                    end
                end

                for i = 1:size(X_test,1)
                    x_cord = X_test(i,featureSeq{1,1});
                    if isempty(featureSeq{2,x_cord})
                        y_cord = 1;
                    else
                        y_cord = X_test(i, featureSeq{2,x_cord});
                    end
                    if iscell(pred{1}{x_cord}) ~= 1
                        y_test(index(i),noOfBaseClassifiers) = pred{1}{x_cord};%y_pred(x_cord, y_cord);
                    else
                        y_test(index(i),noOfBaseClassifiers) = pred{1}{x_cord}{y_cord};
                    end%y_test(i,noOfBaseClassifiers) = y_pred(x_cord, y_cord);
                end

            end

            yt_train = zeros(1, size(index, 2), 1);

            for i = 1:size(X_train,1)
                if sum(y_train(index(i),:)==1)> sum(y_train(index(i),:)==-1)
                    yt_train(i) = 1;
                else
                    yt_train(i) = -1;
                end
            end
            train_error(x, noOfSampleGroup) = sum(yt_train(1,:)'~=X(:,1));

            for i = 1:size(X_test,1)
                if sum(y_test(i,:)==1)> sum(y_test(i,:)==-1)
                    yt_test(i) = 1;
                else
                    yt_test(i) = -1;
                end
            end
            test_error(x, noOfSampleGroup) = sum(yt_test(1,:)'~=X_test(:,1));

        end

        train_error = train_error./training_SampleSize;
        test_error = test_error./test_SampleSize;

        avg_train_error = mean(train_error,2);
        avg_test_error = mean(test_error,2);

        sd_train = std(train_error,0,2);
        sd_test = std(test_error,0,2);

    end

    output = [avg_train_error; avg_test_error; sd_train; sd_test]'
    output2 = [mean(output)]

    plot(B, avg_train_error, 'Color', 'b')
    legend('avgTrainError')
    xlabel('Number of bags');
    ylabel('Average Train Error');
    figure;
    plot(B, avg_test_error, 'Color', 'r')
    legend('avgTestError');
    xlabel('Number of bags');
    ylabel('Average Test Error');
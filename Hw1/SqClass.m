
function [trainError, testError, SD_train, SD_test] = SqClass(filename,num_crossval)

    clearvars;

    %feature vector preparation
    %input_matrix = csvread('G:\Machine Learning (5525)\hw1_data\spam.csv');
    %input_matrix = csvread('G:\Machine Learning (5525)\hw1_data\MNIST-1378.csv');

    %num_crossval = 10;
    
    input_matrix = csvread(filename);
    
    %noOfClasses = no of classes; N = no of samples; F = no of features
    [N, F] = size(input_matrix);
    noOfClasses = length(unique(input_matrix(:,1)));

    X_dash = input_matrix(:,2:end);
    Y = input_matrix(:,1);

    % Formation of the class vector Y
    if noOfClasses == 2
        for i = 1:N
            if(input_matrix(i,1) == 0)
                Y_dash(i,:) = [1 0];
            elseif(input_matrix(i,1) == 1)
                Y_dash(i,:) = [0 1];
            end
        end
    elseif noOfClasses == 4
        for i = 1:N
            if(input_matrix(i,1) == 1)
                Y_dash(i,:) = [1 0 0 0];
            elseif(input_matrix(i,1) == 3)
                Y_dash(i,:) = [0 1 0 0];
            elseif(input_matrix(i,1) == 7)
                Y_dash(i,:) = [0 0 1 0];
            elseif(input_matrix(i,1) == 8)
                Y_dash(i,:) = [0 0 0 1];
            end
        end
    end

    %noOfClasses = no of classes; N = no of samples; F = no of features
    training_SampleSize = N/10;
    for noOfSampleGroup = 1:num_crossval

        X_test = X_dash((noOfSampleGroup-1)*training_SampleSize+1:noOfSampleGroup*training_SampleSize, :);
        Y_test = Y_dash((noOfSampleGroup-1)*training_SampleSize+1:noOfSampleGroup*training_SampleSize, :);

        if noOfSampleGroup == 1
            %xTraining = x(((index*sizeMatrix)+1):m,:);
            X_train = X_dash((noOfSampleGroup*training_SampleSize+1):N, :);
            Y_train = Y_dash((noOfSampleGroup*training_SampleSize+1):N, :);
        else
            X_train = X_dash(1:(noOfSampleGroup-1)*training_SampleSize, :);
            Y_train = Y_dash(1:(noOfSampleGroup-1)*training_SampleSize, :);
            X_train = [X_train;X_dash(noOfSampleGroup*training_SampleSize+1:N, :)];
            Y_train = [Y_train;Y_dash(noOfSampleGroup*training_SampleSize+1:N, :)];
        end

        W = inv(transpose(X_train)*X_train)*transpose(X_train)*Y_train;

        Y_pred_train = transpose(W)*transpose(X_train);%max is taken to take higher of the probabilities of the two classes
        Y_pred_test = transpose(W)*transpose(X_test);

        [a1, Y1] = max(Y_train');
        [a2, Y2] = max(Y_pred_train);
        [a3, Y3] = max(Y_test');
        [a4, Y4] = max(Y_pred_test);


        error_train(noOfSampleGroup) = sum(abs(Y2 - Y1));
        error_test(noOfSampleGroup) = sum(abs(Y4 - Y3));

    end

    trainError = mean(error_train/(N - training_SampleSize))
    testError = mean(error_test/(training_SampleSize))

    SD_train = sqrt(var(error_train/(N - training_SampleSize)))    %train error Standard Deviation
    SD_test = sqrt(var(error_test/training_SampleSize))    %test error Standard Deviation

    %%%%for making random groups for kfold rather than sequential
    %%input_set = ones(1,F+1);
    %%training_SampleSize = round(0.1*N);
    %%for noOfSampleGroup = 1:10
    %%    clear X;
    %%    X = zeros(training_SampleSize,F);
    %%    for i = 1:training_SampleSize
    %%        if (isempty(input_matrix))
    %%            break;
    %%        end
    %%        index = random('unid', length(input_matrix));
    %%        X(i,:) = input_matrix(index,:);
    %%        input_matrix = setdiff(input_matrix, X, 'rows');
    %%    end
    %%    input_set = [input_set;[noOfSampleGroup*ones(training_SampleSize,1) X]];
    %%end


    %%for noOfSampleGroup = 1:1%0

    %%    X_dash = find(input_set(:,1) ~= noOfSampleGroup);
    %%    X = input_set(X_dash,3:F+1);
    %%    Y = input_set(X_dash,2);

    %%    W = inv(transpose(X)*X)*transpose(X)*Y;

    %%    X_test_dash = find(input_set(:,1) == noOfSampleGroup);
    %%    X_test = input_set(X_test_dash,3:F+1);
    %%    Y_test = input_set(X_test_dash,2);

    %%    Y_test_pred = X_test * W;
    %%    error = Y_test - Y_test_pred;

    %%end


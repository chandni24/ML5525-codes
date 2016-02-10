
function [trainError, SD_train, testError,SD_test] = diagFisherLD ( filename, num_crossval )

    %clearvars;

    %feature vector preparation
    %input_matrix = csvread('G:\Machine Learning (5525)\hw1_data\spam.csv');
    %input_matrix = csvread('G:\Machine Learning (5525)\hw1_data\MNIST-1378.csv');
    
    input_matrix = csvread(filename);
    
    num_crossval = 10;

    %noOfClasses = no of classes; N = no of samples; F = no of features
    [N, F] = size(input_matrix);
    noOfClasses = length(unique(input_matrix(:,1)));

    X_dash = input_matrix(:,2:end);
    Y_dash = input_matrix(:,1);

    if noOfClasses == 2
        classVector = [0 1];
    elseif noOfClasses == 4
        classVector = [1 3 7 8];
    end

    %noOfClasses = no of classes; N = no of samples; F = no of features
    training_SampleSize = floor(N/num_crossval);
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

        means = mean(X_train);

        %Grouping classes and computing means over them
        for i = 1:noOfClasses
            index = find(Y_train(:,1) ~= classVector(i));
            X_train_class{i}= X_train(index,:);
            classMean{i} = mean(X_train_class{i});
        end

        %within the class covariance
        cov_W = eye(size((X_train(1) - means)' * (X_train(1) - means)));

        %between the class covariance
        cov_B = zeros(size((classMean{1} - means)' * (classMean{1} - means)));
        for i = 1 : noOfClasses
            cov_B = ((classMean{i} - means)' * (classMean{i} - means)) * size(X_train_class{i}, 1) + cov_B;
        end

        %Calculate W by calulating the eigen Vector
        [eigenVector, eigenValue] = eigs(inv(cov_W) * cov_B);
        W = eigenVector(:, 1 : noOfClasses - 1);

        %new test and train projected data
        new_X_train = X_train * W;
        new_X_test = X_test * W;

        %Grouping classes for computing mean and covariance for the gaussian distribution
        for i = 1:noOfClasses
            index = find(Y_train(:,1) ~= classVector(i));
            new_X_train_class{i}= new_X_train(index,:);
        end

        % Calculate y for the Training Data
        for i = 1 : noOfClasses
            for index = 1 : size(new_X_train, 1)
                new_Y_train{i}(index) = mvnpdf(new_X_train(index, :), mean(new_X_train_class{i}), cov(new_X_train_class{i}));
            end
        end

        for index = 1 : size(transpose(new_Y_train{1}), 1)
            maximumVal = -1;
            maximumInd = 1;
            for i = 1 : noOfClasses
                if new_Y_train{i}(1, index) > maximumVal
                    maximumInd = i;
                    maximumVal = new_Y_train{i}(1, index);
                end
            end
            new_Y2_train(index) = maximumInd;
        end

        % Calculate y for the Training Data
        for i = 1 : noOfClasses
            for index = 1 : size(new_X_test, 1)
                new_Y_test{i}(index) = mvnpdf(new_X_test(index, :), mean(new_X_train_class{i}), cov(new_X_train_class{i}));
            end
        end

        for index = 1 : size(transpose(new_Y_test{1}), 1)
            maximumVal = -1;
            maximumInd = 1;
            for i = 1 : noOfClasses
                if new_Y_test{i}(1, index) > maximumVal
                    maximumInd = i;
                    maximumVal = new_Y_test{i}(1, index);
                end
            end
            new_Y2_test(index) = maximumInd;
        end

        error_train(noOfSampleGroup) = sum(Y_train' ~= classVector(new_Y2_train));
        error_test(noOfSampleGroup) = sum(Y_test' ~= classVector(new_Y2_test));

    end

    %Training Error Mean
    trainError = mean(error_train/(N-training_SampleSize))
    %Training Error Standard Deviation
    SD_train = sqrt(var(error_train/(N-training_SampleSize)))

    %Test Error Mean
    testError = mean(error_test/(training_SampleSize))
    %Test Error Standard Deviation
    SD_test = sqrt(var(error_test/(training_SampleSize)))
    
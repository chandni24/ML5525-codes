
function [testError] = logisticRegression(data_filename,labels_filename,num_Splits,trainingSetPercentage)

    %clearvars;

    %%trainingSetPercentage = [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10];
    %%num_Splits = 5;
    %%input_matrix = csvread('G:\Machine Learning (5525)\hw1_data\data_20newsgroups.csv');
    %%label_matrix = csvread('G:\Machine Learning (5525)\hw1_data\labels_20newsgroups.csv');
    
    input_matrix = csvread(data_filename);
    label_matrix = csvread(labels_filename);

    [M, N] = size(input_matrix);

    %select num_Splits random samples
    for z = 1:num_Splits;

        sampleSize = M*trainingSetPercentage(z)/100;

        for a = 1:10

            clear X
            test_error = 0;

            training_SampleSize = 0.8*sampleSize;
            for i = 1:training_SampleSize
                index = random('unid', length(input_matrix(:,1)));
                X(i,:) = input_matrix(index,:);
                Y(i) = label_matrix(input_matrix(index,1));
            end
            clear i;
            %plot(X, Y, '*');

            %%compute the cost and gradient%%
            [m, n] = size(X);

            % Add intercept term to x and X_test
            X = [ones(m, 1) X];

            % Initialize fitting parameters
            initial_theta = zeros(n + 1, 1);

            %for multiclass classification
            for i = 1:length(unique(Y))
                label_iClass = zeros(1,length(Y));
                for k = 1:length(Y)
                    if Y(k) == i
                        label_iClass(k) = i;
                    else
                        label_iClass(k) = 0;
                    end
                end
                clear k;

                % Compute and display initial cost and gradient
                [cost, gradient] = costFunction(initial_theta, X, label_iClass);


                %%Optimizing the cost function%%
                %  Set options for fminunc
                options = optimset('GradObj', 'on', 'MaxIter', 400);

                %  Run fminunc to obtain the optimal theta
                %  This function will return theta and the cost 
                [theta, cost] = fminunc(@(t)(costFunction(t, X, Y)), initial_theta, options);

                %plotDecisionBoundary(theta, X, Y);

                %test_SampleSize - number of test samples
                test_SampleSize = 0.2*sampleSize;
                for test_counter = 1:test_SampleSize
                    clear X_test;

                    index = random('unid', length(input_matrix(:,1)));
                    X_test = [1 input_matrix(index,:)];
                    Y_test = label_matrix(input_matrix(index,1));
                    H_test = sigmoid(X_test*theta);
                    if (Y_test == test_counter && H_test >= 0.5)%current running class, classified
                        test_error = test_error + 1;
                    elseif (Y_test ~= test_counter && H_test < 0.5)
                        test_error = test_error + 1;
                    end
                end
                clear test_counter;

                testErrorPercentage(i) = test_error/test_SampleSize;

            end %change primary class for classification
            clear i;

            testSampleErrorPercentage(a) = mean(testErrorPercentage);

        end %run every training set 10 times

        testError(z) = mean(testSampleErrorPercentage)
        
        plot(testError, sampleSize, '*');

    end


function [testError] = naiveBayes(data_filename,labels_filename,num_Splits,trainingSetPercentage)

    %clearvars;

    %feature vector preparation with docId as rows, wordID as columns and wordIDFreq in the matrix

    %trainingSetPercentage = [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10];
    %input_matrix = csvread('G:\Machine Learning (5525)\hw1_data\data_20newsgroups.csv');
    %label_matrix = csvread('G:\Machine Learning (5525)\hw1_data\labels_20newsgroups.csv');
    
    input_matrix = csvread(data_filename);
    label_matrix = csvread(labels_filename);
    
    [M, N] = size(input_matrix);

    %select k random samples
    for z = 1:1; %k size random sample

        sampleSize = M*trainingSetPercentage(z)/100;

        for a = 1:1%0

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

            %Finding frequency of each word in each class
            wordLabels = zeros(max(X(:,2)),length(unique(Y)));

            for i = 1:1%length(unique(Y))
                words = zeros(max(X(:,2)),1);
                docID = find(Y(1,:) == i);
                for j = 1:numel(docID)
                    words(X(docID,2)) = words(X(docID,2)) + X(docID,3);
                end
                wordLabels(:,i) = words;
            end

            %prior probablity for class i
            prior = zeros(length(unique(Y)),1);

            for i = 1:length(unique(Y))

                % setting laplacian coefficient = 1
                laplacian = 1;

                prior(i) = (length(find(Y==i)))/length(Y);

            end %change primary class for classification

            %test_SampleSize - number of test samples
            test_SampleSize = 0.2*sampleSize;
            for test_counter = 1:test_SampleSize

                index = random('unid', length(input_matrix(:,1)));
                X_test(test_counter,:) = input_matrix(index,:);
                Y_test(test_counter) = label_matrix(input_matrix(index,1));
            end

            for test_counter = 1:test_SampleSize

                indices = find(X_test(:,1) == test_counter);
                wordIds = X_test(indices,2);

                for j=1:length(unique(Y))
                    %posteriors_test(j) = (wordLabels(wordIds,j) + laplacian)./(sum(wordLabels(:,j)) + (laplacian * length(unique(X(:,2))))) * (prior(j));
                    posteriors_test(i,j) = exp(sum(log((wordLabels(wordIds,j) + laplacian)./(sum(wordLabels(:,j)) + (laplacian * length(unique(X(:,2)))))) + log(prior(j))));
                end
            end

            z = 1:length(test_SampleSize);
            [dummy,posteriors] = max((posteriors_test(z,:)),[],2);

            errors = posteriors(:)-Y_test(:);
            errors(errors == 0) = [];
            no_of_errors = length(errors);

            %test error
            testError = (length(errors)/test_SampleSize*100)

        end %run every training set 10 times

        %        testError(z) = mean(testSampleErrorPercentage)

    end

%filename = 'D:\Machine Learning (5525)\HW2\MNIST-13.csv';
%numruns = 5;
%k = [1 20 100 200 2000];

function w = pegasos(filename, k, numruns)

    global k;

    %initializing variables
    numruns = 5;
    lambda = 0.01;

    input_matrix = csvread(filename);

    [M, N] = size(input_matrix);
    X = input_matrix(:,2:end);
    minimum = min(X,[],1);
    maximum = max(X,[],1);

    %Normalize the input feature vector
    for i = 1:N-1
        for j = 1:M-1
            X(j,i) = (X(j,i))/(maximum(i) - minimum(i));
        end
    end

    %Assign +1 and -1 for the two classes
    Y = input_matrix(:,1);
    Y(find(Y==3)) = -1;

    %k = [1 20 100 200 M];

    for x = 1:size(k,2)
        %Run for different mini-batches
        kValue = k(x);

        for y = 1:numruns
            %5 runs for each mini-batch size

            tStart = tic;

            %w(1) = a vector with norm <= 1/sq_root(lambda)
            w = zeros(1,N-1);

            %for a = 1:length(T)

            T_curr = 10000;
            for t = 1:T_curr

                %Selecting k training samples from both the classes
                ind_class = (find(input_matrix(:,1) == 1));
                ind = datasample(ind_class,floor(kValue/2),'Replace',false);
                ind_class = (find(input_matrix(:,1) == 3));
                ind = cat(1, ind, (datasample(ind_class,ceil(kValue/2),'Replace',false)));

                At = input_matrix(ind,2:end);   %At is equivalent to X
                Yt = input_matrix(ind,1);

                %Find index of training sample with non-zero loss
                index = (Yt.*(At*transpose(w))) < 1;

                %Define the step size for the gradient method
                learningRate = 1/(lambda * t);

                %Update the weight vector
                w_half = ((1 - (learningRate*lambda))*w) + ((learningRate/kValue) * sum(transpose(Yt(index,:))*At(index,:)));
                w = min(1, ((1/sqrt(lambda))/norm(w_half))) * w_half;

                %Compute the objective function value
                loss = max(0, (1 - (Y.*(X*transpose(w)))));
                objFnc(x,t) = ((lambda/2)*(norm(w))^2) + (sum(loss(loss>0))/kValue);

            end

            %Compute the time required for the current iteration of SMO
            tElapsed(y)=toc(tStart);

        end

        %Compute and display the mean and standard deviation for the 5
        %iterations
        avgRuntime(x) = mean(tElapsed)
        SDRuntime(x) = std(tElapsed)

    end

    %Plot the objective function
    plots(objFnc);
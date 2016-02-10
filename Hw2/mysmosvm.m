%filename = 'D:\Machine Learning (5525)\HW2\MNIST-13.csv';
%numruns = 5;

function w = mysmosvm(filename, numruns)

    input_matrix = csvread(filename);

    [M, N] = size(input_matrix);
    global X Y E K eps C L_alpha objF z;

    %initializing variables
    C = 0.1;
    X = input_matrix(:,2:N);
    minimum = min(X,[],1);
    maximum = max(X,[],1);
    
    %Normalize the input feature vector
    X = normc(X);
    %den = maximum - minimum;
    %den(find(den==0)) = 1;
    %for i = 1:N-1
    %    X(:,i) = (X(:,i) - minimum(i))/(den(i));
    %end

    %Assign +1 and -1 for the two classes
    Y = input_matrix(:,1);
    Y(find(Y==3)) = -1;

    %Define a linear kernel
    kernel=@(x,z) x*z';
    %kernel=@(x,z) (x*z'+1)^2;
    K = kernel(X, X);

    eps = 10^(-4);


    for n = 1:numruns

        L_alpha = zeros(M, 1);

        % set error to be the worst, so it will be improved
        E = -Y;

        examineAll = 1;
        numChanged = 0;
        z = 1;

        tStart=tic;

        %Iterate until convergence is not achieved
        while numChanged >0 || examineAll
            numChanged = 0;
            if (examineAll)
                % loop over all training examples
                for i=1:length(Y)
                    numChanged = numChanged + examineExample(i);
                end

            else
                % loop over examples where alpha is not 0 and not C
                for i=1:length(Y)
                    if (L_alpha(i)> 0 && L_alpha(i) < C)
                        numChanged = numChanged + examineExample(i);
                    end
                end
            end

            %Run over full training set and non-bound samples alternately
            if (examineAll == 1)
                examineAll = 0;
            elseif (numChanged == 0)
                examineAll = 1;
            end

        end

        %Compute the objective function
        objFnc{n} = objF;

        %Compute the time required for the current iteration of SMO
        tElapsed(n)=toc(tStart);

    end

    colorstring = 'kbgrm';
    figure(1);
    cla;
    hold on
    for i = 1:numruns
        %plot all the iterations for different values of batches in one graph
        plot(objFnc{i}, 'Color', colorstring(i))
    end

    %Compute and display the mean and standard deviation for the 5
    %iterations
    avg_time = mean(tElapsed)
    std_time = std(tElapsed)

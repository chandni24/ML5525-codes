
function w = plots(objFnc)

    global k;

    colorstring = 'kbgrm';
    figure(1);
    cla;
    hold on
    for i = 1:size(k,2)
        %plot all the iterations for different values of batches in one graph
        plot(objFnc(i,:), 'Color', colorstring(i))
    end
    legend('k = 1', 'k = 20', 'k = 100', 'k = 200', 'k = 2000');

    hold off
    for i = 1:size(k,2)
        figure;
        %plot different curves for different batch sizes
        plot(objFnc(i,:));
    end

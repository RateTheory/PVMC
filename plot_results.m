%% Script for plotting figures

% PVMC Recovered Matrix
f=figure();
plot(Xvmc2', '-o')
ylabel('Matrix Elements (colored by row)')
xlabel('Column')
title('CF3CH3 Recovered Matrix')

% Ground Truth Matrix
f = figure();
plot(Xtrue', '-o')
ylabel('Matrix Elements (colored by row)')
xlabel('Column')
title('CF3CH3 True Matrix')

% Influence of Each Gradient Term over the PVMC Iterations
f = figure();
semilogy(iter_info.gradterms')
ylabel('Matrix Norm')
xlabel('Iteration')
title('Gradient Terms')
legend('Rank Minimization Term', 'X-QS')

% Comparison to a one-shot Polynomial Interpolation
f = figure();
plot(Xvmc2', 'o')
hold on
set(gca,'ColorOrderIndex',1)
plot((Qi*S)')
ylabel('Matrix Elements (colored by row)')
xlabel('Column')
title('Recovered Matrix (o) vs. Polynomial Interpolation of Sampled Columns (-)')
%legend('PVMC Recovered Matrix', 'Polynomial Interpolation')

% Plot Error vs iteration
f = figure();
plot(error2)
ylabel('Error')
xlabel('Iteration')
title('Error over iteration')

% Overlay PVMC matrix onto ground truth matrix
f = figure();
plot(Xtrue', 'o-')
hold on
%plot((Qi*S)','-')
set(gca,'ColorOrderIndex',1)
plot(Xvmc2', '+')
ylabel('Matrix Elements (colored by row)')
xlabel('Column')
title('Comparison of True (o) and PVMC (+) Matrices')
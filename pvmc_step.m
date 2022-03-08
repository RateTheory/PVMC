function [X,error,error_qs, Q_init, Q,S,iter_info]=pvmc_step(Xsamp,Xtrue,sampmask,I,s, options)
    %Jeongmin Chae and Stephen Quiton, University of Southern California, 2022

    %options: d,p,tau,iterations, exit tol
    d = options.d;
    p = options.p;
    iter = options.niter;
    exit_tol = options.exit_tol;
    polydegree = options.polynomial_degree;
    lambda = options.lambda;
    gammamin = options.gammamin;
    %tau = options.tau;

    %Construct Q and S based on degree and sampled columns
    [Q,S] = row_interpolation(Xtrue,polydegree,I,s);

    Q_init = Q;
    Xpredictpoly = Q*S;

    scalefac = sqrt(max(sum(abs(Xsamp).^2,'all')));
    scalefacQ = sqrt(max(sum(abs(Q).^2,'all')));
    scalefacS=sqrt(max(sum(abs(S).^2,'all')));


    Xtemp = Xsamp/scalefac; %normalize data;
    X=Xtemp;
    Xsamp = Xtemp;
    Xtrue=Xtrue/scalefac;
    Q=Q/scalefacQ;
    S=S/scalefacS;
    Xpredictpoly = Xpredictpoly/scalefac;

    q = 1-(p/2);
    eta=1.01; %1.01
    M = zeros(size(X,1),size(S,2));

    Xold = X;

    for i=1:iter

        Q_iter{i} = Q;
        if i~=1
             Q = X*S'*inv(S*S');
        end

        %% Kernel-eig
        G=X'*X;

        c=1;
        K = (G+c).^d;
        [V,D] = eig(K);
        [ev,idx] = sort(abs(diag(D)),'descend');
        V = V(:,idx);
        if i==1 % || i==2
            gamma = 0.01*ev(1);
        end
        evinv = (ev+gamma).^(-q);
        E = diag(evinv);
        W = V*E*V';


        if d == 1
            rank_term = X*W;
        elseif d == 2
            rank_term = 2*X*(W.*(G+c));
        elseif d > 2 && d < Inf
            rank_term = d*X*(W.*((G+c).^(d-1)));
        end


        gradX = (1-lambda)*rank_term+lambda*(X-Q*S);
        %gradX = 2*X*(W.*(G+c))+(X-Q*S);

        gradterms(1,i) = norm((1-lambda)*rank_term, 'fro');
        gradterms(2,i) = norm(lambda*(X-Q*S), 'fro');
        XW_iter{i} = X*W;
        W_iter{i} = W;
        tau = gamma^q;

        X = X - tau*gradX;

        gamma=gamma/eta;
        gamma = max(gamma,gammamin);
        X(sampmask)=0; %Gets reset here...
        Xw = Xsamp+X;

        X=Xw;


        error(i) = norm(X-Xtrue,'fro')/norm(Xtrue,'fro');
        error_qs(i) = norm(Q*S-Xtrue,'fro')/norm(Xtrue,'fro');

        X_iter{i} = X;



        % check for convergence
        update(i) = norm(X-Xold,'fro')/norm(Xold,'fro');
        if( update(i) < exit_tol )
        fprintf('PVMC reached exit tolerance at iter %d\n',i);
            break;
        end
        Xold = X;
    end

    X=X*scalefac;
    S=S*scalefacS;
    Q = X*S'*inv(S*S');


    iter_info.gradterms = gradterms;
    iter_info.XW = XW_iter;
    iter_info.X = X_iter;
    iter_info.W = W_iter;
    iter_info.Q = Q_iter;

end


function [Q,S] = row_interpolation(X,degree,I,s)
    % Create new QS based on cone-sampled columns
    S=zeros(degree+1,size(X,2));

    % Get initial S
    for i=1:size(S,1)
        for j=1:size(S,2)
            S(i,j)=s(j).^(size(S,1)-i);
        end
    end

    Sfit = S(:,I);
    X = X(:,I);
    Q = X*Sfit'*inv(Sfit*Sfit');
end

% RBF SVM : K = exp(-gamma||x1-x2||^2)
clc
clear
load('train.mat');
load('eval.mat');

%% Form the evaluation set

% load('test.mat');
% combined_data = horzcat(train_data, test_data);
% combined_label = horzcat(train_label', test_label');
% eval_indice = randperm(size(combined_data,2),600);
% eval_data = combined_data(:, eval_indice);
% eval_label = combined_label(eval_indice);
% eval_label = eval_label';
% save('eval.mat','eval_data','eval_label')

%% preprocess the data
train_mean = mean(train_data,2);
eval_mean = mean(eval_data,2);
std_train = std(train_data, 0, 2);
std_eval = std(eval_data, 0, 2);
norm_train_data = zeros(57,2000);
norm_eval_data = zeros(57,1536);
for i = 1:size(train_data,2)
    norm_train_data(:,i) = (train_data(:,i) - train_mean)./std_train;
end

for i = 1:size(eval_data,2)
    norm_eval_data(:,i) = (eval_data(:,i) - eval_mean)./std_eval;
end

%% compute the Gram matrix, H matrix and check the admissible
gram_matrix = zeros(2000,2000);
H = zeros(2000,2000);

% soft margin with polynomial kernel
for sigma = 1:5
    gamma = 1/(57 * sigma^2);
    
    for i = 1:size(norm_train_data,2)
        for j = 1:size(norm_train_data,2)
            gram_matrix(i,j) = exp(-1 * gamma * (norm(norm_train_data(:,i) - norm_train_data(:,j)))^2 );
            H(i,j) = train_label(i) * train_label(j) * gram_matrix(i,j);
        end
    end
    
    eig_value = eig(gram_matrix);
    if min(eig_value) < -1e-4
        disp('kernel is not admissible')
    else
        disp('kernel is admissible')
    end
    
    %% alpha calculation
    for C = [0.1, 0.6, 1.1, 2.1]
        
        f = -ones(2000,1);
        
        A = [];
        b = [];
        
        Aeq = train_label';
        beq = 0;
        
        lb = zeros(2000,1);
        ub = ones(2000,1)*C;
        
        x0 = [];
        options = optimset('LargeScale','off','MaxIter', 1000);
        alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
    
        %% select support vector
        threshold = 1e-4;
            sv_idx = find(alpha > threshold & alpha < C);
            
            %% Solve discriminant variables
            b = zeros(size(sv_idx));
            for i = 1:size(sv_idx)
            wx = 0;
                for j = 1:size(norm_train_data,2)
                    wx = wx + alpha(j,1) * train_label(j) * exp(-1 * gamma * (norm(norm_train_data(:,i) - norm_train_data(:,j)))^2 );
                end
                b(i) = 1./train_label(sv_idx(i)) - wx';
            end
            b0 = mean(b);
            
            %% test accuracy and training accuracy
            pred_train = zeros(2000,1);
            pred_train_label = zeros(2000,1);
            for i = 1:2000
                for j = 1:2000
                    pred_train(i,1) = pred_train(i,1) + alpha(j,1) * train_label(j) * exp(-1 * gamma * (norm(norm_train_data(:,i) - norm_train_data(:,j)))^2 );   
                end
                pred_train(i,1) = pred_train(i,1) + b0;
              
                if pred_train(i,1) > 0
                    pred_train_label(i,1) = 1;
                else
                    pred_train_label(i,1) = -1;
                end
            end
            pred_train_acc = (sum(abs(pred_train_label-train_label)))/2;
            pred_train_acc = 1 - pred_train_acc/2000;
            
            pred_eval = zeros(600,1);
            pred_eval_label = zeros(600,1);
            for i = 1:600
                for j = 1:2000
                    pred_eval(i,1) = pred_eval(i,1) + alpha(j,1) * train_label(j) * exp(-1 * gamma * (norm(norm_train_data(:,j) - norm_eval_data(:,i)))^2 ); 
           
                end
                pred_eval(i,1) = pred_eval(i,1) + b0;
                if pred_eval(i,1) > 0
                    pred_eval_label(i,1) = 1;
                else
                    pred_eval_label(i,1) = -1;
                end
            end
            pred_eval_acc = (sum(abs(pred_eval_label-eval_label)))/2;
            pred_eval_acc = 1 - pred_eval_acc/600;
            disp(['sigma = ' num2str(sigma) ', C = ' num2str(C)]);
            disp(['training accuracy: ' num2str(pred_train_acc) ', Evaluation accuracy: ' num2str(pred_eval_acc)]);
    end        
end


% Soft margin SVM with polynomial kernel: K = (x1'x2+1)^p, 0 <= alpha <= C
clc
clear
load('train.mat');
load('test.mat');

%% preprocess the data
train_mean = mean(train_data,2);
test_mean = mean(test_data,2);
std_train = std(train_data, 0, 2);
std_test = std(test_data, 0, 2);
norm_train_data = zeros(57,2000);
norm_test_data = zeros(57,1536);
for i = 1:size(train_data,2)
    norm_train_data(:,i) = (train_data(:,i) - train_mean)./std_train;
end

for i = 1:size(test_data,2)
    norm_test_data(:,i) = (test_data(:,i) - test_mean)./std_test;
end

%% compute the Gram matrix, H matrix and check the admissible
gram_matrix = zeros(2000,2000);
H = zeros(2000,2000);

% soft margin with polynomial kernel
for p = 1:5
    for i = 1:size(norm_train_data,2)
        for j = 1:size(norm_train_data,2)
            gram_matrix(i,j) = (norm_train_data(:,i)' * norm_train_data(:,j) + 1)^p;
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
                    wx = wx + alpha(j,1) * train_label(j) * (norm_train_data(:,j)' * norm_train_data(:,sv_idx(i))+1)^p;
                end
                b(i) = 1./train_label(sv_idx(i)) - wx';
            end
            b0 = mean(b);
            
            %% test accuracy and training accuracy
            pred_train = zeros(2000,1);
            pred_train_label = zeros(2000,1);
            for i = 1:2000
                for j = 1:2000
                    pred_train(i,1) = pred_train(i,1) + alpha(j,1) * train_label(j) * (norm_train_data(:,j)' * norm_train_data(:,i)+1)^p; 
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
            
            pred_test = zeros(1536,1);
            pred_test_label = zeros(1536,1);
            for i = 1:1536
                for j = 1:2000
                    pred_test(i,1) = pred_test(i,1) + alpha(j,1) * train_label(j) * (norm_train_data(:,j)' * norm_test_data(:,i)+1)^p; 
                end
                pred_test(i,1) = pred_test(i,1) + b0;
                if pred_test(i,1) > 0
                    pred_test_label(i,1) = 1;
                else
                    pred_test_label(i,1) = -1;
                end
            end
            pred_test_acc = (sum(abs(pred_test_label-test_label)))/2;
            pred_test_acc = 1 - pred_test_acc/1536;
            disp(['p = ' num2str(p) ', C = ' num2str(C), ', b0 = ', num2str(b0)]);
            disp(['Training accuracy: ' num2str(pred_train_acc) ', Testing accuracy: ' num2str(pred_test_acc)]);
    end        
end
    
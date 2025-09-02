clear all;
clc;

% load('encoder4.mat','dlnetEncoder')
% load decoder4.mat
% load layer4.mat

load('encoder4.mat','dlnetEncoder')
load decoder4.mat
load layer4.mat
rng(2)

load traintest3.mat
mintrain = min(train);
maxtrain = max(train);
% load daxiao.mat
% mintrain = train_max;
% maxtrain = train_min;

test = train(1:end,1:3);
% test = train(2251:end,1:3);
true = test;
% true1 = train(1:2250,1:3);
true1 = train(1:2250,1:3);
train = train(1:2250,1:3);

Nsim1 = length(test)-1;

for i=1:3
train(:,i) = (train(:,i)-mintrain(i))/(maxtrain(i)-mintrain(i));
test(:,i) = (test(:,i)-mintrain(i))/(maxtrain(i)-mintrain(i));
end

U1 = test(1:end-1,1)';
X = train(1:end-1,2:end)';
Y = train(2:end,2:end)';
U =  train(1:end-1,1)';

n = 2;
m = 1; % number of control inputs
basisFunction = 'rbf';
% RBF centers
Nrbf =0   ;
cent = rand(n,Nrbf)*2 -1;
rbf_type = 'thinplate'; %thinplate,gauss,invquad,invmultquad,polyharmonic
% Lifting mapping - RBFs + the state itself
liftFun = @(xx)( [xx;rbf(xx,cent,rbf_type)] );
Nlift = Nrbf + n  ;
Xlift = liftFun(X);
Ylift = liftFun(Y);

W1 = [Ylift ; X];
V1 = [Xlift; U];
VVt1 = V1*V1';
WVt1 = W1*V1';
M1 = WVt1 * pinv(VVt1); % Matrix [A B; C 0]
Alift = M1(1:Nlift,1:Nlift);
Blift = M1(1:Nlift,Nlift+1:end);
Clift = M1(Nlift+1:end,1:Nlift);
test0 = liftFun(test(1,2:end)');
for i = 1:Nsim1
    test0 = [test0, Alift*test0(:,end) + Blift*U1(i)]; % Lifted dynamics
    
end
test0 = test0(1:2,:);
for i=1:2
test0(i,:) = test0(i,:)*(maxtrain(i+1)-mintrain(i+1))+mintrain(i+1);
end

lw = 1;
figure
plot(test0(1,:),'linewidth',lw); hold on
plot(true(:,2),'linewidth',lw);

figure
plot(test0(2,:),'linewidth',lw); hold on
plot(true(:,3),'linewidth',lw);
legend('真实值', 'koopman方法预测值');

zuixiao = test0';

% ##############
n = 2;
m = 1; % number of control inputs
basisFunction = 'rbf';
% RBF centers
Nrbf =5   ;
cent = rand(n,Nrbf)*2 -1;
rbf_type = 'thinplate'; %thinplate,gauss,invquad,invmultquad,polyharmonic
% Lifting mapping - RBFs + the state itself
liftFun = @(xx)( [xx;rbf(xx,cent,rbf_type)] );
Nlift = Nrbf + n  ;
Xlift = liftFun(X);
Ylift = liftFun(Y);

W1 = [Ylift ; X];
V1 = [Xlift; U];
VVt1 = V1*V1';
WVt1 = W1*V1';
M1 = WVt1 * pinv(VVt1); % Matrix [A B; C 0]
Alift = M1(1:Nlift,1:Nlift);
Blift = M1(1:Nlift,Nlift+1:end);
Clift = M1(Nlift+1:end,1:Nlift);
test0 = liftFun(test(1,2:end)');
for i = 1:Nsim1
    test0 = [test0, Alift*test0(:,end) + Blift*U1(i)]; % Lifted dynamics
    
end
test0 = test0(1:2,:);
for i=1:2
test0(i,:) = test0(i,:)*(maxtrain(i+1)-mintrain(i+1))+mintrain(i+1);
end

lw = 1;
figure
plot(test0(1,:),'linewidth',lw); hold on
plot(true(:,2),'linewidth',lw);

figure
plot(test0(2,:),'linewidth',lw); hold on
plot(true(:,3),'linewidth',lw);
legend('真实值', 'koopman方法预测值');
kooprbf = test0';

load("traintest5.mat");
result = [];
order=20;
u=train(:,1);
figure;
subplot(2,2,1);
y=train(:,2);
result = ARXEst(u,y,order,result);
subplot(2,2,2);
y=train(:,3);
result = ARXEst(u,y,order,result);

arxx = result;
tic;
AB = dlnetlayer.Layers(2,1).Weights;
train_min = min(train(1:end,:));
train_max = max(train(1:end,:));
train(:,1) = (train(:,1)-train_min(:,1))/(train_max(:,1)-train_min(:,1));
train(:,2) = (train(:,2)-train_min(:,2))/(train_max(:,2)-train_min(:,2));
train(:,3) = (train(:,3)-train_min(:,3))/(train_max(:,3)-train_min(:,3));
u1 = train(1:end,1)';
data = dlarray(train(:,1:3)','CB');
u = dlarray(train(:,1)','CB');
g = forward(dlnetEncoder, data(2:3,:));
encoderData = extractdata(g(:,1:end));
encoderu = extractdata(u);
X = [encoderData(:,1:end-1);encoderu(:,1:end-1)];
Y = encoderData(:,2:end);



x_koop2 = X(:,1);
X_koop2 = x_koop2(1:end-1,:);
for i=1:4501-1 
    x_koop2 = AB*x_koop2;
    X_koop2 = [X_koop2,x_koop2];
    x_koop2 = [x_koop2;u1(:,i+1)];
end

result = forward(dlnetDecoder,[X_koop2;dlarray(u1(1,1:end),'CB')]);
result= extractdata(result)';
result(:,1) = (result(:,1)*(train_max(:,2)-train_min(:,2)))+train_min(:,2);
result(:,2) = (result(:,2)*(train_max(:,3)-train_min(:,3)))+train_min(:,3);
figure
plot(result(:,1),'linewidth',lw); hold on
plot(true1(:,2),'linewidth',lw);
toc;
x_koop1 = X(:,1);
X_koop1 = x_koop1(1:end-1,:);
for i=1:4501-1 
    x_koop1 = AB*x_koop1;
    X_koop1 = [X_koop1,x_koop1];
    x_koop1 = [x_koop1;u1(:,i+1)];
end

result_test = forward(dlnetDecoder,[X_koop1;dlarray(u1(1,1:end),'CB')]);
result_test = extractdata(result_test)';
result_test(:,1) = (result_test(:,1)*(train_max(:,2)-train_min(:,2)))+train_min(:,2);
result_test(:,2) = (result_test(:,2)*(train_max(:,3)-train_min(:,3)))+train_min(:,3);
figure
plot(result_test(:,1),'linewidth',lw); hold on
plot(true(:,2),'linewidth',lw);

figure
plot(result_test(:,2),'linewidth',lw); hold on
plot(true(:,3),'linewidth',lw);
legend('真实值', 'koopman方法预测值');



% load gpr1.mat
% [result_pre1,~,yint1] = predict(gprMdl1,result_test(1:end-1,:));
% [result_pre2,~,yint2] = predict(gprMdl2,result_test(1:end-1,:));
% 
% 
% tmp = result_test(1:end-1,:)+[result_pre1,result_pre2];
% result_test=[result_test(1,:); tmp];

load traintest_g_result1.mat
% load traintest_result1_new.mat
 load ab_JT9D.mat
koopdp =  [result_test(1,:); winer_koop];
% koopdp =  result_test;
% result_test = [result_test(1,:); winer_koopg];
% load dgp_JT9D1.mat
% load dgp_JT9D2.mat

% load traintest_g_result2.mat
result_test = [result_test(1,:); winer_koopg];

load recur_JT9D.mat
JT9D1=recur(:,1);
JT9D2=recur(:,2);
load dynonet_JT9D1.mat
load dynonet_JT9D2.mat

dynonet1= dynonet1*(maxtrain(2)-mintrain(2))+mintrain(2);
dynonet2= dynonet2*(maxtrain(3)-mintrain(3))+mintrain(3);

koopgauss = result_test;

% for i=1:2
% test0(i,:) = test0(i,:)*(maxtrain(i+1)-mintrain(i+1))+mintrain(i+1);
% end

figure
t = 0:0.04:180 - 99*0.04; 
% 计算误差的绝对值
e_rkp = abs(kooprbf(100:end,1) - true(100:end,2));
e_arx = abs(arxx(100:end,1) - true(100:end,2));
e_dkp = abs(koopdp(100:end,1) - true(100:end,2));
e_ddynonet = abs(dynonet1(100:end,1) - true(100:end,2));
e_drecur = abs(JT9D1(100:end,1) - true(100:end,2));
e_dkissgpr = abs(koopgauss(100:end,1) - true(100:end,2));

% 绘图
plot(t, e_rkp, 'b', 'linewidth', lw); hold on
plot(t, e_arx, 'g', 'linewidth', lw); hold on
plot(t, e_dkp, 'c', 'linewidth', lw); hold on
plot(t, e_ddynonet, 'm', 'linewidth', lw); hold on  % Changed to magenta dashed
plot(t, e_drecur, 'Color',[1,0.5,0.2], 'linewidth', lw); hold on  % Changed to black dashed
plot(t, e_dkissgpr, 'r', 'linewidth', lw + 0.5); hold on

xlabel('Time(s)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('|Error| (rpm)', 'FontSize', 16, 'FontWeight', 'bold');
xlim([0 180]);
legend('RKP','ARX','DKP','dynoNet','RecReg','DKIGPR');

figure
t = 0:0.04:180 - 99*0.04; 
% 计算误差的绝对值
e_rkp = abs(kooprbf(100:end,2) - true(100:end,3));
e_arx = abs(arxx(100:end,2) - true(100:end,3));
e_dkp = abs(koopdp(100:end,2) - true(100:end,3));
e_ddynonet = abs(dynonet2(100:end,1) - true(100:end,3));
e_drecur = abs(JT9D2(100:end,1) - true(100:end,3));
e_dkissgpr = abs(koopgauss(100:end,2) - true(100:end,3));

% 绘图
plot(t, e_rkp, 'b', 'linewidth', lw); hold on
plot(t, e_arx, 'g', 'linewidth', lw); hold on
plot(t, e_dkp, 'c', 'linewidth', lw); hold on
plot(t, e_ddynonet, 'm', 'linewidth', lw); hold on  % Changed to magenta dashed
plot(t, e_drecur, 'Color',[1,0.5,0.2], 'linewidth', lw); hold on  % Changed to black dashed
plot(t, e_dkissgpr, 'r', 'linewidth', lw + 0.5); hold on

xlabel('Time(s)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('|Error| (rpm)', 'FontSize', 16, 'FontWeight', 'bold');
xlim([0 180]);
legend('RKP','ARX','DKP','dynoNet','RecReg','DKIGPR');




figure
% plot(zuixiao(:,2),'linewidth',lw); hold on
t = 0:0.04:180-99*0.04; 
h1=plot(t,true(100:end,3),'--k','linewidth',lw+0.5);hold on
h2=plot(t,kooprbf(100:end,2),'b','linewidth',lw); hold on
h3=plot(t,arxx(100:end,2),'g','linewidth',lw); hold on

h4=plot(t,koopdp(100:end,2),'c','linewidth',lw); hold on
h_dynonet =plot(t, dynonet2(100:end,1), 'm', 'linewidth', lw); hold on  % Changed to magenta dashed
h_recur= plot(t,JT9D2(100:end,1), 'Color',[1,0.5,0.2], 'linewidth', lw); hold on  % Changed to black dashed
h5=plot(t,koopgauss(100:end,2),'r','linewidth',lw+0.5); hold on
xlabel('Time(s)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('NH(rpm)', 'FontSize', 16, 'FontWeight', 'bold');
xlim([0 180]);



legend([h1, h2, h3, h4, h_dynonet,h_recur,h5], ...
       {'TRUE','RKP','ARX','DKP','dynoNet','RecReg','DKIGPR'});

% 5) 最后把 TRUE 这条线推到最上层
uistack(h1, 'top');

figure
% plot(zuixiao(:,2),'linewidth',lw); hold on
h1=plot(t, true(100:end,2), '--k', 'linewidth', lw+0.5); 
hold on

% 中间线条：使用默认颜色（不指定颜色参数）
h2=plot(t, kooprbf(100:end,1),'b', 'linewidth', lw); 
hold on
h3=plot(t, arxx(100:end,1),'g', 'linewidth', lw); 
hold on
h4=plot(t, koopdp(100:end,1),'c', 'linewidth', lw); 
hold on
h_dynonet =plot(t, dynonet1(100:end,1), 'm', 'linewidth', lw); hold on  % Changed to magenta dashed
h_recur= plot(t,JT9D1(100:end,1), 'Color',[1,0.5,0.2], 'linewidth', lw); hold on  % Changed to black dashed
% 最后一条线：红色（'r' 表示红色）
h5=plot(t, koopgauss(100:end,1), 'r', 'linewidth', lw+0.5); 
xlabel('Time(s)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('NL(rpm)', 'FontSize', 16, 'FontWeight', 'bold');
xlim([0 180]);


legend([h1, h2, h3, h4, h_dynonet,h_recur,h5], ...
       {'TRUE','RKP','ARX','DKP','dynoNet','RecReg','DKIGPR'});

% 5) 最后把 TRUE 这条线推到最上层
uistack(h1, 'top');



errzuixiao = sqrt(mse(zuixiao(100:end,2),true(100:end,3)));
errrbf =  sqrt(mse(kooprbf(100:end,2),true(100:end,3)));
errgauss = sqrt(mse(koopgauss(100:end,2),true(100:end,3)));
errdp = sqrt(mse(koopdp(100:end,2),true(100:end,3)));
errarxx = sqrt(mse(arxx(100:end,2),true(100:end,3)));
errdynonet = sqrt(mse(dynonet2(100:end,1), true(100:end,3)));  % 修正语法错误
errcur = sqrt(mse(JT9D2(100:end,1), true(100:end,3)));        % 修正语法错误


% 提取数据子集
true_vals = true(100:end,3);
zuixiao_vals = zuixiao(100:end,2);
rbf_vals = kooprbf(100:end,2);
gauss_vals = koopgauss(100:end,2);
dp_vals = koopdp(100:end,2);
arxx_vals = arxx(100:end,2);
dynonet_vals = dynonet2(100:end,1);
cur_vals = JT9D2(100:end,1);

% 计算MAPE
mape_zuixiao = mean(abs((true_vals - zuixiao_vals) ./ true_vals)) * 100;
mape_rbf = mean(abs((true_vals - rbf_vals) ./ true_vals)) * 100;
mape_gauss = mean(abs((true_vals - gauss_vals) ./ true_vals)) * 100;
mape_dp = mean(abs((true_vals - dp_vals) ./ true_vals)) * 100;
mape_arxx = mean(abs((true_vals - arxx_vals) ./ true_vals)) * 100;
mape_dynonet = mean(abs((true_vals - dynonet_vals) ./ true_vals)) * 100;
mape_cur = mean(abs((true_vals - cur_vals) ./ true_vals)) * 100;

% 计算MSLE (使用log1p避免log(0)问题)
msle_zuixiao = mean((log1p(zuixiao_vals) - log1p(true_vals)).^2);
msle_rbf = mean((log1p(rbf_vals) - log1p(true_vals)).^2);
msle_gauss = mean((log1p(gauss_vals) - log1p(true_vals)).^2);
msle_dp = mean((log1p(dp_vals) - log1p(true_vals)).^2);
msle_arxx = mean((log1p(arxx_vals) - log1p(true_vals)).^2);
msle_dynonet = mean((log1p(dynonet_vals) - log1p(true_vals)).^2);
msle_cur = mean((log1p(cur_vals) - log1p(true_vals)).^2);


errzuixiao1 = sqrt(mse(zuixiao(100:end,1),true(100:end,2)));
errrbf1 =  sqrt(mse(kooprbf(100:end,1),true(100:end,2)));
errgauss1 = sqrt(mse(koopgauss(100:end,1),true(100:end,2)));
errdp1 = sqrt(mse(koopdp(100:end,1),true(100:end,2)));
errarxx1 = sqrt(mse(arxx(100:end,1),true(100:end,2)));
errdynonet1 = sqrt(mse(dynonet1(100:end,1), true(100:end,2)));  % 修正语法错误
errcur1 = sqrt(mse(JT9D1(100:end,1), true(100:end,2)));        % 修正语法错误


naive_forecast_3 = true(99:end-1, 3); % 朴素预测：使用前一个时间步的值
actual_values_3 = true(100:end, 3); % 对应的实际值
baseline_mae_3 = mean(abs(actual_values_3 - naive_forecast_3));

% 对于第二组数据（第2列）
naive_forecast_2 = true(99:end-1, 2); % 朴素预测：使用前一个时间步的值
actual_values_2 = true(100:end, 2); % 对应的实际值
baseline_mae_2 = mean(abs(actual_values_2 - naive_forecast_2));

% 现在计算各个模型的MASE（第一组：第3列数据）
mase_min_3 = mase(zuixiao(100:end,2), true(100:end,3), baseline_mae_3);
mase_rbf_3 = mase(kooprbf(100:end,2), true(100:end,3), baseline_mae_3);
mase_gauss_3 = mase(koopgauss(100:end,2), true(100:end,3), baseline_mae_3);
mase_dp_3 = mase(koopdp(100:end,2), true(100:end,3), baseline_mae_3);
mase_arx_3 = mase(arxx(100:end,2), true(100:end,3), baseline_mae_3);
mase_dynonet_3 = mase(dynonet2(100:end,1), true(100:end,3), baseline_mae_3);
mase_cur_3 = mase(JT9D2(100:end,1), true(100:end,3), baseline_mae_3);

% 计算各个模型的MASE（第二组：第2列数据）
mase_min_2 = mase(zuixiao(100:end,1), true(100:end,2), baseline_mae_2);
mase_rbf_2 = mase(kooprbf(100:end,1), true(100:end,2), baseline_mae_2);
mase_gauss_2 = mase(koopgauss(100:end,1), true(100:end,2), baseline_mae_2);
mase_dp_2 = mase(koopdp(100:end,1), true(100:end,2), baseline_mae_2);
mase_arx_2 = mase(arxx(100:end,1), true(100:end,2), baseline_mae_2);
mase_dynonet_2 = mase(dynonet1(100:end,1), true(100:end,2), baseline_mae_2);
mase_cur_2 = mase(JT9D1(100:end,1), true(100:end,2), baseline_mae_2);


% 提取数据子集
true_vals1 = true(100:end,2);
zuixiao_vals1 = zuixiao(100:end,1);
rbf_vals1 = kooprbf(100:end,1);
gauss_vals1 = koopgauss(100:end,1);
dp_vals1 = koopdp(100:end,1);
arxx_vals1 = arxx(100:end,1);
dynonet_vals1 = dynonet1(100:end,1);
cur_vals1 = JT9D1(100:end,1);

% 计算MAPE
mape_zuixiao1 = mean(abs((true_vals1 - zuixiao_vals1) ./ true_vals1)) * 100;
mape_rbf1 = mean(abs((true_vals1 - rbf_vals1) ./ true_vals1)) * 100;
mape_gauss1 = mean(abs((true_vals1 - gauss_vals1) ./ true_vals1)) * 100;
mape_dp1 = mean(abs((true_vals1 - dp_vals1) ./ true_vals1)) * 100;
mape_arxx1 = mean(abs((true_vals1 - arxx_vals1) ./ true_vals1)) * 100;
mape_dynonet1 = mean(abs((true_vals1 - dynonet_vals1) ./ true_vals1)) * 100;
mape_cur1 = mean(abs((true_vals1 - cur_vals1) ./ true_vals1)) * 100;

% 计算MSLE (使用log1p避免log(0)问题)
msle_zuixiao1 = mean((log1p(zuixiao_vals1) - log1p(true_vals1)).^2);
msle_rbf1 = mean((log1p(rbf_vals1) - log1p(true_vals1)).^2);
msle_gauss1 = mean((log1p(gauss_vals1) - log1p(true_vals1)).^2);
msle_dp1 = mean((log1p(dp_vals1) - log1p(true_vals1)).^2);
msle_arxx1 = mean((log1p(arxx_vals1) - log1p(true_vals1)).^2);
msle_dynonet1 = mean((log1p(dynonet_vals1) - log1p(true_vals1)).^2);
msle_cur1 = mean((log1p(cur_vals1) - log1p(true_vals1)).^2);


y_true = true(100:end,3);
y_range = max(y_true) - min(y_true);  % 计算真实值的范围[1,2](@ref)mnb

errzuixiao3 = sqrt(mse(zuixiao(100:end,2), y_true)) / y_range;
errrbf3 = sqrt(mse(kooprbf(100:end,2), y_true)) / y_range;
errgauss3 = sqrt(mse(koopgauss(100:end,2), y_true)) / y_range;
errdp3 = sqrt(mse(koopdp(100:end,2), y_true)) / y_range;
errarxx3 = sqrt(mse(arxx(100:end,2), y_true)) / y_range;
% errsi3 = sqrt(mse(y_encoder(100:end,1), y_true)) / y_range;
errdynonet3 = sqrt(mse(dynonet2(100:end,1), y_true))/ y_range;  % 修正语法错误
errcur3 = sqrt(mse(JT9D2(100:end,1), y_true))/ y_range;        % 修正语法错误

% 第二组误差（基于true的第2列）
y_true1 = true(1:end,2);
y_range1 = max(y_true1) - min(y_true1);  % 新真实值范围[1,4](@ref)

errzuixiao4 = sqrt(mse(zuixiao(1:end,1), y_true1)) / y_range1;
errrbf4 = sqrt(mse(kooprbf(1:end,1), y_true1)) / y_range1;
errgauss4 = sqrt(mse(koopgauss(1:end,1), y_true1)) / y_range1;
errdp4 = sqrt(mse(koopdp(1:end,1), y_true1)) / y_range1;
errarxx4 = sqrt(mse(arxx(1:end,1), y_true1)) / y_range1;
% errsi4 = sqrt(mse(y_encoder1(1:end,1), y_true1))/ y_range1 ;
errdynonet4 = sqrt(mse(dynonet1(1:end,1), y_true1))/ y_range1;  % 修正语法错误
errcur4 = sqrt(mse(JT9D1(1:end,1), y_true1))/ y_range1;        % 修正语法错误



TEzuixiao = calculateERR(zuixiao(:,1),true(:,2));
TErbf = calculateERR(kooprbf(:,1),true(:,2));
TEarxx = calculateERR(arxx(:,1),true(:,2));
TEgauss = calculateERR(koopgauss(1:end,1),true(1:end,2));
TEdp = calculateERR(koopdp(1:end,1),true(1:end,2));

TEzuixiao1 = calculateERR(zuixiao(:,2),true(:,3));
TErbf1 = calculateERR(kooprbf(:,2),true(:,3));
TEarxx1 = calculateERR(arxx(:,2),true(:,3));
TEgauss1 = calculateERR(koopgauss(1:end,2),true(1:end,3));
TEdp1 = calculateERR(koopdp(1:end,2),true(1:end,3));


R2zuixiao = computeR2_multidim(zuixiao(1:end,:),true(1:end,2:3));
R2rbf = computeR2_multidim(kooprbf(1:end,:),true(1:end,2:3));
R2arxx = computeR2_multidim(arxx(1:end,:),true(1:end,2:3));
R2gauss = computeR2_multidim(koopgauss(1:end,:), true(1:end,2:3));
R2dp = computeR2_multidim(koopdp(1:end,:), true(1:end,2:3));
R2dynonet= computeR2_multidim([dynonet1,dynonet2], true(1:end,2:3));
R2cur = computeR2_multidim([JT9D1,JT9D2], true(1:end,2:3));

function result = ARXEst(u,y,order,result)
umin=min(u);umax=max(u);
u1=(u-umin)/(umax-umin);

ymin=min(y);ymax=max(y);
y1=(y-ymin)/(ymax-ymin);

% y1=normalize(y,'range');
% u1=normalize(u,'range');
% y1=y;u1=u;

output_name = 'y1';
input_name = 'u1';
names = {output_name,input_name};

output_lag = [1   ];
input_lag = [1 2 3 ];
lags = {output_lag,input_lag};

lreg = linearRegressor(names,lags);

% z=iddata(y1(1:2250),u1(1:2250));
% m = nlarx(z,lreg);
% ysim1 = sim(m,u1(2250:end));

z=iddata(y1,u1);
model = arx(z, [2 2 1]);
compare(model,z);
ysim1 = sim(model,u1(1:end));

y1 = y1*(ymax-ymin)+ymin;
ysim1 = ysim1*(ymax-ymin)+ymin;
result = [result,ysim1];
plot(y1(1:end),'r');
hold on;
plot(ysim1(1:end),'b');
hold off;
end

function mase_value = mase(forecast, actual, baseline_mae)
    % forecast: 预测值
    % actual: 实际值
    % baseline_mae: 基准预测的MAE
    
    % 确保向量长度一致
    if length(forecast) ~= length(actual)
        error('预测值和实际值的长度必须一致');
    end
    
    % 计算当前预测的MAE
    forecast_mae = mean(abs(actual - forecast));
    
    % 计算MASE
    mase_value = forecast_mae / baseline_mae;
end
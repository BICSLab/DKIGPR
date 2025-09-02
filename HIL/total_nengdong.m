clear all;
clc;

load zuixin2.mat


test = train(2002:8000,:);
true = test;

train = train(2000:6000,:);

Nsim1 = length(test)-1;

train_min = min(train(1:end,:));
train_max = max(train(1:end,:));
train(:,1) = (train(:,1)-train_min(:,1))/(train_max(:,1)-train_min(:,1));
train(:,2) = (train(:,2)-train_min(:,2))/(train_max(:,2)-train_min(:,2));
train(:,3) = (train(:,3)-train_min(:,3))/(train_max(:,3)-train_min(:,3));
train(:,4) = (train(:,4)-train_min(:,4))/(train_max(:,4)-train_min(:,4));

test(:,1) = (test(:,1)-train_min(:,1))/(train_max(:,1)-train_min(:,1));
test(:,2) = (test(:,2)-train_min(:,2))/(train_max(:,2)-train_min(:,2));
test(:,3) = (test(:,3)-train_min(:,3))/(train_max(:,3)-train_min(:,3));
test(:,4) = (test(:,4)-train_min(:,4))/(train_max(:,4)-train_min(:,4));


U1 = test(1:end-1,1:2)';
X = train(1:end-1,3:end)';
Y = train(2:end,3:end)';
U =  train(1:end-1,1:2)';

n = 2;
m = 2; % number of control inputs
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
test0 = liftFun(test(1,3:end)');
for i = 1:Nsim1

    test0 = [test0, Alift*test0(:,end) + Blift*U1(:,i)]; % Lifted dynamics
    
end
test0 = test0(1:2,:);

test0(1,:) = (test0(1,:)*(train_max(:,3)-train_min(:,3)))+train_min(:,3);
test0(2,:) = (test0(2,:)*(train_max(:,4)-train_min(:,4)))+train_min(:,4);

lw = 1;
figure
plot(test0(1,:),'linewidth',lw); hold on
plot(true(:,3),'linewidth',lw);


legend( 'koopman方法预测值','真实值');

zuixiao = test0';

% ##############
n = 2;
m = 2; % number of control inputs
basisFunction = 'rbf';
% RBF centers
Nrbf =1  ;
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
test0 = liftFun(test(1,3:end)');
for i = 1:Nsim1
    test0 = [test0, Alift*test0(:,end) + Blift*U1(:,i)]; % Lifted dynamics
    
end
test0 = test0(1:2,:);
test0(1,:) = (test0(1,:)*(train_max(:,3)-train_min(:,3)))+train_min(:,3);
test0(2,:) = (test0(2,:)*(train_max(:,4)-train_min(:,4)))+train_min(:,4);

lw = 1;
figure
plot(test0(1,:),'linewidth',lw); hold on
plot(true(:,3),'linewidth',lw);


legend( 'koopman方法预测值','真实值');
kooprbf = test0';
test(:,1) = test(:,1)*(train_max(:,1)-train_min(:,1))+train_min(:,2);
test(:,2) = test(:,2)*(train_max(:,2)-train_min(:,2))+train_min(:,2);
test(:,3) = test(:,3)*(train_max(:,3)-train_min(:,3))+train_min(:,3);
test(:,4) = test(:,4)*(train_max(:,4)-train_min(:,4))+train_min(:,4);

result = [];
order=20;
u=test(:,1:2);
figure;
subplot(2,2,1);
y=test(:,3);
result = ARXEst(u,y,order,result);
subplot(2,2,2);
y=test(:,4);
result = ARXEst(u,y,order,result);

arxx = result;

% figure
% plot(zuixiao(2:end,1),'linewidth',lw); hold on
% plot(kooprbf(2:end,1),'linewidth',lw); hold on
% plot(arxx(2:end,1),'linewidth',lw); hold on
% 
% 
% plot(true(2:end,2),'linewidth',lw);
% legend('最小二乘','koopmanrbf','arxx', '真实');
% 
% errzuixiao = mse(zuixiao(:,1),true(:,2));
% errrbf =  mse(kooprbf(:,1),true(:,2));
% errarxx = mse(arxx(:,1),true(:,2));

load nengdong_g_result2.mat
load nengdong_result2.mat
% load ab_nengdong.mat
% load HIL1.mat
load dynonet_HIL0.mat
load dynonet_HIL1.mat
load recur_HIL.mat
% load dgp_HIL0.mat
% load dgp_HIL1.mat

t = 0:0.005:30-3*0.005; 
figure

% 计算误差
e_rkp = abs(kooprbf(2:end,1) - true(2:end,3));
e_arx = abs(arxx(2:end,1) - true(2:end,3));
e_dkp = abs(winer_koop(:,1) - true(2:end,3));
e_ddynonet = abs(dynonet0(2:end,1) - true(2:end,3));
e_drecur = abs(recur(2:end,1) - true(2:end,3));
e_dkissgpr = abs(winer_koopg(:,1) - true(2:end,3));

% 绘图
plot(t, e_rkp, 'b', 'linewidth', lw); hold on
plot(t, e_arx, 'g', 'linewidth', lw); hold on
plot(t, e_dkp, 'c', 'linewidth', lw); hold on
plot(t, e_ddynonet, 'm', 'linewidth', lw); hold on  % Changed to magenta dashed
plot(t, e_drecur, 'Color',[1,0.5,0.2], 'linewidth', lw); hold on  % Changed to black dashed
plot(t, e_dkissgpr, 'r', 'linewidth', lw+0.5); hold on

xlabel('Time(s)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('|Error| (rpm)', 'FontSize', 16, 'FontWeight', 'bold');
xlim([0 30]);
legend('RKP','ARX','DKP','dynoNet','RecReg','DKIGPR');


figure

% 计算误差
e_rkp = abs(kooprbf(2:end,2) - true(2:end,4));
e_arx = abs(arxx(2:end,2) - true(2:end,4));
e_dkp = abs(winer_koop(:,2) - true(2:end,4));
e_ddynonet = abs(dynonet1(2:end,1) - true(2:end,4));
e_drecur = abs(recur(2:end,2) - true(2:end,4));
e_dkissgpr = abs(winer_koopg(:,2) - true(2:end,4));

% 绘图
plot(t, e_rkp, 'b', 'linewidth', lw); hold on
plot(t, e_arx, 'g', 'linewidth', lw); hold on
plot(t, e_dkp, 'c', 'linewidth', lw); hold on
plot(t, e_ddynonet, 'm', 'linewidth', lw); hold on  % Changed to magenta dashed
plot(t, e_drecur, 'Color',[1,0.5,0.2], 'linewidth', lw); hold on  % Changed to black dashed
plot(t, e_dkissgpr, 'r', 'linewidth', lw+0.5); hold on

xlabel('Time(s)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('|Error|', 'FontSize', 16, 'FontWeight', 'bold');
xlim([0 30]);
legend('RKP','ARX','DKP','dynoNet','RecReg','DKIGPR');



t = 0:0.005:30-3*0.005; 
figure
% plot(zuixiao(2:end,1),'linewidth',lw); hold on
h1=plot(t,true(2:end,3),'--k','linewidth',lw+0.5);hold on
h2=plot(t,kooprbf(2:end,1),'b','linewidth',lw); hold on
h3=plot(t,arxx(2:end,1),'g','linewidth',lw); hold on

h4=plot(t,winer_koop(:,1),'c','linewidth',lw);hold on 

h_dynonet =plot(t, dynonet0(2:end,1), 'm', 'linewidth', lw); hold on  % Changed to magenta dashed
h_recur= plot(t,recur(2:end,1), 'Color',[1,0.5,0.2], 'linewidth', lw); hold on  % Changed to black dashed
h5=plot(t,winer_koopg(:,1),'r','linewidth',lw+0.5); hold on


xlabel('Time(s)', 'FontSize', 16, 'FontWeight', 'bold');
% ylabel('π', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('n_2(rpm)', 'FontSize', 16, 'FontWeight', 'bold');
xlim([0 30]);
legend([h1, h2, h3, h4, h_dynonet,h_recur,h5], ...
       {'TRUE','RKP','ARX','DKP','dynoNet','RecReg','DKIGPR'});

% 5) 最后把 TRUE 这条线推到最上层
uistack(h1, 'top');


figure
% plot(zuixiao(2:end,1),'linewidth',lw); hold on
h1=plot(t,true(2:end,4),'--k','linewidth',lw+0.5);hold on
h2=plot(t,kooprbf(2:end,2),'b','linewidth',lw); hold on
h3=plot(t,arxx(2:end,2),'g','linewidth',lw); hold on

h4=plot(t,winer_koop(:,2),'c','linewidth',lw);hold on 
h_dynonet =plot(t, dynonet1(2:end,1), 'm', 'linewidth', lw); hold on  % Changed to magenta dashed
h_recur= plot(t,recur(2:end,2), 'Color',[1,0.5,0.2], 'linewidth', lw); hold on  % Changed to black dashed
h5=plot(t,winer_koopg(:,2),'r','linewidth',lw+0.5); hold on



xlabel('Time(s)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('π', 'FontSize', 16, 'FontWeight', 'bold');
% ylabel('n_2(rpm)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0 30]);
legend([h1, h2, h3, h4, h_dynonet,h_recur,h5], ...
       {'TRUE','RKP','ARX','DKP','dynoNet','RecReg','DKIGPR'});

% 5) 最后把 TRUE 这条线推到最上层
uistack(h1, 'top');

errzuixiao = sqrt(mse(zuixiao(:,1),true(:,3)));
errrbf =  sqrt(mse(kooprbf(:,1),true(:,3)));
errarxx = sqrt(mse(arxx(:,1),true(:,3)));
errdynoNet = sqrt(mse(dynonet0(:,1),true(:,3)));
errcur=sqrt(mse(recur(1:end,1) , true(1:end,3)));
errkoopg = sqrt(mse(winer_koopg(:,1),true(2:end,3)));
errkoop = sqrt(mse(winer_koop(:,1),true(2:end,3)));

errzuixiao1 = sqrt(mse(zuixiao(:,2),true(:,4)));
errrbf1 =  sqrt(mse(kooprbf(:,2),true(:,4)));
errarxx1 = sqrt(mse(arxx(:,2),true(:,4)));
errdynoNet1 = sqrt(mse(dynonet1(:,1),true(:,4)));
errcur1=sqrt(mse(recur(1:end,2) , true(1:end,4)));
errkoopg1 = sqrt(mse(winer_koopg(:,2),true(2:end,4)));
errkoop1 = sqrt(mse(winer_koop(:,2),true(2:end,4)));


naive_forecast_3 = true(1:end-1, 3); % 朴素预测：使用前一个时间步的值
actual_values_3 = true(2:end, 3); % 对应的实际值
baseline_mae_3 = mean(abs(actual_values_3 - naive_forecast_3));

% 对于第二组数据（第4列）
naive_forecast_4 = true(1:end-1, 4); % 朴素预测：使用前一个时间步的值
actual_values_4 = true(2:end, 4); % 对应的实际值
baseline_mae_4 = mean(abs(actual_values_4 - naive_forecast_4));

% 现在计算各个模型的MASE（第一组：第3列数据）
mase_min_3 = mase(zuixiao(:,1), true(:,3), baseline_mae_3);
mase_rbf_3 = mase(kooprbf(:,1), true(:,3), baseline_mae_3);
mase_arx_3 = mase(arxx(:,1), true(:,3), baseline_mae_3);
mase_dynonet_3 = mase(dynonet0(:,1), true(:,3), baseline_mae_3);
mase_cur_3 = mase(recur(1:end,1), true(1:end,3), baseline_mae_3);
mase_koop_g_3 = mase(winer_koopg(:,1), true(2:end,3), baseline_mae_3);
mase_koop_3 = mase(winer_koop(:,1), true(2:end,3), baseline_mae_3);

% 计算各个模型的MASE（第二组：第4列数据）
mase_min_4 = mase(zuixiao(:,2), true(:,4), baseline_mae_4);
mase_rbf_4 = mase(kooprbf(:,2), true(:,4), baseline_mae_4);
mase_arx_4 = mase(arxx(:,2), true(:,4), baseline_mae_4);
mase_dynonet_4 = mase(dynonet1(:,1), true(:,4), baseline_mae_4);
mase_cur_4 = mase(recur(1:end,2), true(1:end,4), baseline_mae_4);
mase_koop_g_4 = mase(winer_koopg(:,2), true(2:end,4), baseline_mae_4);
mase_koop_4 = mase(winer_koop(:,2), true(2:end,4), baseline_mae_4);

% 第一个输出变量（第3列）的评估
true_vals3 = true(:,3);

% 确保维度匹配
true_zuixiao3 = true_vals3(1:length(zuixiao(:,1)));
true_rbf3 = true_vals3(1:length(kooprbf(:,1)));
true_arxx3 = true_vals3(1:length(arxx(:,1)));
true_dynonet3 = true_vals3(1:length(dynonet0(:,1)));
true_recur3 = true_vals3(1:length(recur(:,1)));
true_koopg3 = true_vals3(2:2+length(winer_koopg(:,1))-1);
true_koop3 = true_vals3(2:2+length(winer_koop(:,1))-1);

% 计算第一个输出变量的MAPE和MSLE
mape_zuixiao3 = safe_mape(true_zuixiao3, zuixiao(:,1));
mape_rbf3 = safe_mape(true_rbf3, kooprbf(:,1));
mape_arxx3 = safe_mape(true_arxx3, arxx(:,1));
mape_dynonet3 = safe_mape(true_dynonet3, dynonet0(:,1));
mape_recur3 = safe_mape(true_recur3, recur(:,1));
mape_koopg3 = safe_mape(true_koopg3, winer_koopg(:,1));
mape_koop3 = safe_mape(true_koop3, winer_koop(:,1));

msle_zuixiao3 = mean((log1p(zuixiao(:,1)) - log1p(true_zuixiao3)).^2);
msle_rbf3 = mean((log1p(kooprbf(:,1)) - log1p(true_rbf3)).^2);
msle_arxx3 = mean((log1p(arxx(:,1)) - log1p(true_arxx3)).^2);
msle_dynonet3 = mean((log1p(dynonet0(:,1)) - log1p(true_dynonet3)).^2);
msle_recur3 = mean((log1p(recur(:,1)) - log1p(true_recur3)).^2);
msle_koopg3 = mean((log1p(winer_koopg(:,1)) - log1p(true_koopg3)).^2);
msle_koop3 = mean((log1p(winer_koop(:,1)) - log1p(true_koop3)).^2);

% 第二个输出变量（第4列）的评估
true_vals4 = true(:,4);

% 确保维度匹配
true_zuixiao4 = true_vals4(1:length(zuixiao(:,2)));
true_rbf4 = true_vals4(1:length(kooprbf(:,2)));
true_arxx4 = true_vals4(1:length(arxx(:,2)));
true_dynonet4 = true_vals4(1:length(dynonet1(:,1)));
true_recur4 = true_vals4(1:length(recur(:,2)));
true_koopg4 = true_vals4(2:2+length(winer_koopg(:,2))-1);
true_koop4 = true_vals4(2:2+length(winer_koop(:,2))-1);

% 计算第二个输出变量的MAPE和MSLE
mape_zuixiao4 = safe_mape(true_zuixiao4, zuixiao(:,2));
mape_rbf4 = safe_mape(true_rbf4, kooprbf(:,2));
mape_arxx4 = safe_mape(true_arxx4, arxx(:,2));
mape_dynonet4 = safe_mape(true_dynonet4, dynonet1(:,1));
mape_recur4 = safe_mape(true_recur4, recur(:,2));
mape_koopg4 = safe_mape(true_koopg4, winer_koopg(:,2));
mape_koop4 = safe_mape(true_koop4, winer_koop(:,2));

msle_zuixiao4 = mean((log1p(zuixiao(:,2)) - log1p(true_zuixiao4)).^2);
msle_rbf4 = mean((log1p(kooprbf(:,2)) - log1p(true_rbf4)).^2);
msle_arxx4 = mean((log1p(arxx(:,2)) - log1p(true_arxx4)).^2);
msle_dynonet4 = mean((log1p(dynonet1(:,1)) - log1p(true_dynonet4)).^2);
msle_recur4 = mean((log1p(recur(:,2)) - log1p(true_recur4)).^2);
msle_koopg4 = mean((log1p(winer_koopg(:,2)) - log1p(true_koopg4)).^2);
msle_koop4 = mean((log1p(winer_koop(:,2)) - log1p(true_koop4)).^2);

% 第一组误差（基于true的第3列）
y_true3 = true(:,3);
y_range3 = max(y_true3) - min(y_true3);  % 计算真实值范围[1,5](@ref)

errzuixiao3 = sqrt(mse(zuixiao(:,1), y_true3)) / y_range3;
errrbf3 = sqrt(mse(kooprbf(:,1), y_true3)) / y_range3;
errarxx3 = sqrt(mse(arxx(:,1), y_true3)) / y_range3;
errkoopg3 = sqrt(mse(winer_koopg(:,1), true(2:end,3))) / (max(true(2:end,3)) - min(true(2:end,3)));
errkoop3 = sqrt(mse(winer_koop(:,1), true(2:end,3))) / (max(true(2:end,3)) - min(true(2:end,3)));
errdynonet3 = sqrt(mse(dynonet0, y_true3)) / y_range3;
errcur3 = sqrt(mse(recur(1:end,1), y_true3)) / y_range3;
% errsi3 = sqrt(mse(y_encoder1(:,1), y_true3)) / y_range3;

% 第二组误差（基于true的第4列）
y_true4 = true(:,4);
y_range4 = max(y_true4) - min(y_true4);  % 新真实值范围[1,6](@ref)

errzuixiao4 = sqrt(mse(zuixiao(:,2), y_true4)) / y_range4;
errrbf4 = sqrt(mse(kooprbf(:,2), y_true4)) / y_range4;
errarxx4 = sqrt(mse(arxx(:,2), y_true4)) / y_range4;
errkoopg4 = sqrt(mse(winer_koopg(:,2), true(2:end,4))) / (max(true(2:end,4)) - min(true(2:end,4)));
errkoop4 = sqrt(mse(winer_koop(:,2), true(2:end,4))) / (max(true(2:end,4)) - min(true(2:end,4)));
errdynonet4 = sqrt(mse(dynonet1, y_true4)) / y_range4;
errcur4 = sqrt(mse(recur(1:end,2), y_true4)) / y_range4;
% errsi4 = sqrt(mse(y_encoder1(:,2), y_true4)) / y_range4;

TEzuixiao = calculateERR(zuixiao(:,1),true(:,3));
TErbf = calculateERR(kooprbf(:,1),true(:,3));
TEarxx = calculateERR(arxx(:,1),true(:,3));
TEgauss = calculateERR(winer_koopg(:,1),true(2:end,3));
TEdp = calculateERR(winer_koop(:,1),true(2:end,3));

TEzuixiao1 = calculateERR(zuixiao(:,2),true(:,4));
TErbf1 = calculateERR(kooprbf(:,2),true(:,4));
TEarxx1 = calculateERR(arxx(:,2),true(:,4));
TEgauss1 = calculateERR(winer_koopg(:,2),true(2:end,4));
TEdp1 = calculateERR(winer_koop(:,2),true(2:end,4));



R2zuixiao = computeR2_multidim(zuixiao(:,1:2),true(:,3:4));
R2rbf = computeR2_multidim(kooprbf(:,1:2),true(:,3:4));
R2arxx = computeR2_multidim(arxx(:,1:2),true(:,3:4));
R2gauss = computeR2_multidim(winer_koopg(:,1:2), true(2:end,3:4));
R2dp = computeR2_multidim(winer_koop(:,1:2), true(2:end,3:4));
R2dynonet = computeR2_multidim([dynonet0,dynonet1], true(1:end,3:4));
R2cur = computeR2_multidim(recur, true(1:end,3:4));

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

ysim1 = sim(model,u1);

y1 = y1*(ymax-ymin)+ymin;
ysim1 = ysim1*(ymax-ymin)+ymin;
result = [result,ysim1];

end

% 安全MAPE计算函数
function mape_val = safe_mape(true_data, pred_data)
    valid_idx = true_data ~= 0 & ~isnan(true_data) & ~isinf(true_data) & ...
                ~isnan(pred_data) & ~isinf(pred_data);
    if sum(valid_idx) == 0
        mape_val = NaN;
    else
        mape_val = mean(abs((true_data(valid_idx) - pred_data(valid_idx)) ./ ...
                      true_data(valid_idx))) * 100;
    end
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
    forecast_mae = mean(abs(actual - forecast),"all");
    
    % 计算MASE
    mase_value = forecast_mae / baseline_mae;
end
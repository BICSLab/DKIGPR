clear all;
clc;

load Silverbox.mat
load silverbox20_noise.mat

train = [U,Y];
% train(:,2)=noise20;
test = [U_val,Y_val];
true = test;
baoliu=train;
train_min = min(train(1:end,:));
train_max = max(train(1:end,:));

Nsim1 = length(test)-1;



U1 = test(1:end-1,1)';
X = train(1:end-1,2:end)';
Y = train(2:end,2:end)';
U =  train(1:end-1,1)';

n = 1;
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
test0 = test0(1,:);


lw = 1;
figure
plot(test0(1,:),'linewidth',lw); hold on
plot(true(:,2),'linewidth',lw);


legend('真实值', 'koopman方法预测值');

zuixiao = test0';

% ##############
n = 1;
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
test0 = test0(1,:);


lw = 1;


figure
plot(test0(1,:),'linewidth',lw); hold on
plot(true(:,2),'linewidth',lw);


legend('真实值', 'koopman方法预测值');
kooprbf = test0';


result = [];
order=20;
u=test(:,1);
figure;
subplot(2,2,1);
y=test(:,2);
u2=baoliu(:,1);
y2=baoliu(:,2);
result = ARXEst(u,y,u2,y2,order,result);


arxx = result;
% load ab_silverbox.mat
% load Silverbox_g_result1.mat
load Silverbox_result1.mat
% load W_result2_noise_20.mat
% load W_result2_noise_20.mat
load Wienerhammer_g_result3_noise.mat

load recur_silverbox.mat
load dynonet_silverbox.mat
recur=recur';
recur(:,1) = recur(:,1)*(train_max(:,2)-train_min(:,2))+train_min(:,2);

% load deepSI.mat
% load deepencoder2.mat

figure
% 计算误差的绝对值
e_rkp = abs(kooprbf(2:end,1) - true(2:end,2));
e_arx = abs(arxx(2:end,1) - true(2:end,2));
e_dkp = abs(winer_koop' - true(2:end,2));
e_ddynonet = abs(dynonet(2:end,1) - true(2:end,2));
e_drecur = abs(recur(2:end,1) - true(2:end,2));
e_dkissgpr =abs(winer_koopg - true(2:end,2));
% e_si = abs(y_pred - true(1:end,2));




% 绘制误差图
plot(e_rkp, 'b', 'linewidth', lw); hold on
plot(e_arx, 'g', 'linewidth', lw); hold on
plot(e_dkp, 'c', 'linewidth', lw); hold on
plot(e_ddynonet, 'm', 'linewidth', lw); hold on  % Changed to magenta dashed
plot(e_drecur, 'Color',[1,0.5,0.2], 'linewidth', lw); hold on  % Changed to black dashed
plot(e_dkissgpr, 'r', 'linewidth', lw+0.5); hold on

xlabel('Time(sample)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('|Error| (V)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0 1023]);
ylim([0 0.5]); % 可根据实际误差幅值调整
legend('RKP','ARX','DKP','dynoNet','RecReg','DKIGPR');


figure
% plot(zuixiao(2:end,1),'linewidth',lw); hold on
h1=plot(true(2:end,2),'--k','linewidth',lw+0.5);hold on
h2=plot(kooprbf(2:end,1),'b','linewidth',lw); hold on
h3=plot(arxx(2:end,1),'g','linewidth',lw); hold on
h4=plot(winer_koop','c','linewidth',lw);hold on 
h_dynonet =plot( dynonet(2:end,1), 'm', 'linewidth', lw); hold on  % Changed to magenta dashed
h_recur= plot(recur(2:end,1), 'Color',[1,0.5,0.2], 'linewidth', lw); hold on  % Changed to black dashed
h5=plot(winer_koopg,'r','linewidth',lw+0.5); hold on

xlabel('Time(sample)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Amplitude(V)', 'FontSize', 16, 'FontWeight', 'bold');
xlim([0 1023]);
ylim([-0.4 0.6]); % 修改这里的数值以适应你的数据范围

legend([h1, h2, h3, h4, h_dynonet,h_recur,h5], ...
       {'TRUE','RKP','ARX','DKP','dynoNet','RecReg','DKIGPR'});

% 5) 最后把 TRUE 这条线推到最上层
uistack(h1, 'top');




errzuixiao = sqrt(mse(zuixiao(:,1),true(:,2)));
errrbf =  sqrt(mse(kooprbf(:,1),true(:,2)));
errarxx = sqrt(mse(arxx(:,1),true(:,2)));
errkoopg = sqrt(mse(winer_koopg,true(2:end,2)));
errkoop = sqrt(mse(winer_koop',true(2:end,2)));
errrecur = sqrt(mse(recur,true(1:end,2)));
errdynonet = sqrt(mse(dynonet,true(1:end,2)));

naive_forecast = true(1:end-1, 2); % 朴素预测：使用前一个时间步的值
actual_values = true(2:end, 2); % 对应的实际值

% 计算基准预测的MAE
baseline_mae = mean(abs(actual_values - naive_forecast));

% 现在计算各个模型的MASE
mase_koop_rbf = mase(kooprbf(2:end,1), true(2:end,2), baseline_mae);
mase_arx = mase(arxx(2:end,1), true(2:end,2), baseline_mae);
mase_koop = mase(winer_koop', true(2:end,2), baseline_mae);
mase_dynonet = mase(dynonet(2:end,1), true(2:end,2), baseline_mae);
mase_recurrent = mase(recur(2:end,1), true(2:end,2), baseline_mae);
mase_koop_gpr = mase(winer_koopg, true(2:end,2), baseline_mae);


teeee=true(1:end,2) - dynonet;

mape_zuixiao = mean(abs((true(:,2) - zuixiao(:,1)) ./ true(:,2))) ;
mape_rbf = mean(abs((true(:,2) - kooprbf(:,1)) ./ true(:,2))) ;
mape_arxx = mean(abs((true(:,2) - arxx(:,1)) ./ true(:,2))) ;
mape_koopg = mean(abs((true(2:end,2) - winer_koopg) ./ true(2:end,2))) ;
mape_koop = mean(abs((true(2:end,2) - winer_koop') ./ true(2:end,2))) ;
mape_recur = mean(abs((true(1:end,2) - recur) ./ true(1:end,2))) ;
mape_dynonet = mean(abs((true(1:end,2) - dynonet) ./ true(1:end,2))) ;

% MSLE (均方对数误差) 计算
msle_zuixiao = mean((log1p(zuixiao(:,1)) - log1p(true(:,2))).^2);
msle_rbf = mean((log1p(kooprbf(:,1)) - log1p(true(:,2))).^2);
msle_arxx = mean((log1p(arxx(:,1)) - log1p(true(:,2))).^2);
msle_koopg = mean((log1p(winer_koopg) - log1p(true(2:end,2))).^2);
msle_koop = mean((log1p(winer_koop') - log1p(true(2:end,2))).^2);
msle_recur = mean((log1p(recur) - log1p(true(1:end,2))).^2);
msle_dynonet = mean((log1p(dynonet) - log1p(true(1:end,2))).^2);


true_full = true(:,2);
range_full = max(true_full) - min(true_full);

% 截断真实值的归一化因子（适用于后两个模型）
true_subset = true(2:end,2);
range_subset = max(true_subset) - min(true_subset);

%% 各模型NRMSE计算
% 注意：需确保预测值与真实值的维度匹配（行向量或列向量）
errzuixiao1 = sqrt(mse(zuixiao(:,1), true_full)) / range_full;         % [7,8](@ref)
errrbf1     = sqrt(mse(kooprbf(:,1), true_full)) / range_full;         % [7,8](@ref)
errarxx1    = sqrt(mse(arxx(:,1), true_full)) / range_full;            % [7,8](@ref)
errkoopg1   = sqrt(mse(winer_koopg(:), true_subset)) / range_subset;   % [7,8](@ref)
errkoop1    = sqrt(mse(winer_koop(:), true_subset)) / range_subset;   %
errrecur1    = sqrt(mse(recur, true_full)) / range_full; 
errdynonet1    = sqrt(mse(dynonet , true_full)) / range_full; 

TEzuixiao = calculateERR(zuixiao(:,1),true(:,2));
TErbf = calculateERR(kooprbf(:,1),true(:,2));
TEarxx = calculateERR(arxx(:,1),true(:,2));
TEgauss = calculateERR(winer_koopg, true(2:end,2));
TEdp = calculateERR(winer_koop', true(2:end,2));


R2zuixiao = computeR2_multidim(zuixiao(:,1),true(:,2));
R2rbf = computeR2_multidim(kooprbf(:,1),true(:,2));
R2arxx = computeR2_multidim(arxx(:,1),true(:,2));
R2gauss = computeR2_multidim(winer_koopg, true(2:end,2));
R2dp = computeR2_multidim(winer_koop', true(2:end,2));
R2recur = computeR2_multidim(recur, true(1:end,2));
R2dgynonet = computeR2_multidim(dynonet , true(1:end,2));

function result = ARXEst(u,y,u2,y2,order,result)
umin=min(u2);umax=max(u2);
u1=(u-umin)/(umax-umin);
u2=(u2-umin)/(umax-umin);

ymin=min(y);ymax=max(y);
y1=(y-ymin)/(ymax-ymin);
y2=(y2-ymin)/(ymax-ymin);

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

z=iddata(y2,u2);
model = arx(z, [2 2 1]);
compare(model,z);

ysim1 = sim(model,u1);

y1 = y1*(ymax-ymin)+ymin;
ysim1 = ysim1*(ymax-ymin)+ymin;
result = [result,ysim1];

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
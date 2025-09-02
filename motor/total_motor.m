clear all;
clc;

load motore.mat
load motor20_noise.mat

test = train;
true = test;

load motorda.mat


train=train(1:1500,:);
% train(:,2:3) = train(:,2:3)+noise10;
train(:,2:3) = train(:,2:3)
baoliu=train;
Nsim1 = length(test)-1;



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


lw = 1;
figure
plot(test0(1,:),'linewidth',lw); hold on
plot(true(:,2),'linewidth',lw);

figure
plot(test0(2,:),'linewidth',lw); hold on
plot(true(:,3),'linewidth',lw);
legend('真实值', 'koopman方法预测值');
kooprbf = test0';

result = [];
order=20;
u=test(:,1);
figure;
subplot(2,2,1);
y=test(:,3);
u2=baoliu(:,1);
y2=baoliu(:,3);
result = ARXEst(u,y,u2,y2,order,result);
arxx=result(1:100,:)

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


load motor_g_result2.mat
load motor_result2.mat
% load motor_g_result2_20.mat
% load motor_result2_20.mat
% load ab_motor.mat
load recur_motor.mat;
load dynonet_motor.mat;

recur=recur';
recur=recur(1:100,:);
dynonet=dynonet(1:100,:);

% figure
% t = 0:0.05:5-0.1;  % 时间轴，对应数据长度应为 100（2:end）
% 
% % 计算误差绝对值ab_motor.matab_motor.mat
% e_rkp = abs(kooprbf(2:end,1) - true(2:end,2));
% e_arx = abs(arxx(2:end,1) - true(2:end,2));
% e_dkp = abs(winer_koop' - true(2:end,2));
% e_dkissgpr = abs(winer_koopg' - true(2:end,2));
% 
% % 画图
% plot(t, e_rkp, 'b', 'linewidth', lw); hold on
% plot(t, e_arx, 'g', 'linewidth', lw); hold on
% plot(t, e_dkp, 'c', 'linewidth', lw); hold on
% plot(t, e_dkissgpr, 'r', 'linewidth', lw+0.5); hold on
% 
% xlim([0 5]);
% ylim([0 0.5]);  % 可根据实际误差幅度调整
% 
% xlabel('Time(s)', 'FontSize', 12, 'FontWeight', 'bold');
% ylabel('|Error|', 'FontSize', 12, 'FontWeight', 'bold');
% 
% legend('RKP Error','ARX Error','DKP Error','DKISSGPR Error');


figure
t = 0:0.05:5-0.1;  % 时间轴，对应数据长度应为 100（2:end）

% 计算误差绝对值
e_rkp = abs(kooprbf(2:end,2) - true(2:end,3));
e_arx = abs(arxx(2:end,1) - true(2:end,3));
e_dkp = abs(winer_koop' - true(2:end,3));
e_ddynonet = abs(dynonet(2:end,1) - true(2:end,3));
e_drecur = abs(recur(2:end,1) - true(2:end,3));
e_dkissgpr = abs(winer_koopg' - true(2:end,3));


% 画图
plot(t, e_rkp, 'b', 'linewidth', lw); hold on
plot(t, e_arx, 'g', 'linewidth', lw); hold on
plot(t, e_dkp, 'c', 'linewidth', lw); hold on
plot(t, e_ddynonet, 'm', 'linewidth', lw); hold on  % Changed to magenta dashed
plot(t, e_drecur, 'Color',[1,0.5,0.2], 'linewidth', lw); hold on  % Changed to black dashed
plot(t, e_dkissgpr, 'r', 'linewidth', lw+0.5); hold on

xlim([0 5]);
ylim([0 0.1]);  % 可根据实际误差幅度调整

xlabel('Time(s)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('|Error|', 'FontSize', 16, 'FontWeight', 'bold');

legend('RKP ','ARX ','DKP ','dynoNet ','RecReg','DKIGPR ');

% figure
% t = 0:0.05:5-0.1; 
% %plot(zuixiao(2:end,1),'linewidth',lw); hold on
% plot(t,true(2:end,2),'k','linewidth',lw+0.5);hold on
% plot(t,kooprbf(2:end,1),'b','linewidth',lw); hold on
% plot(t,arxx(2:end,1),'g','linewidth',lw); hold on
% 
% plot(t,winer_koop','c','linewidth',lw);hold on 
% plot(t,winer_koopg,'r','linewidth',lw+0.5); hold on
% xlim([0 5]);
% ylim([-0.1 0.8]); % 修改这里的数值以适应你的数据范围
% 
% xlabel('Time(s)', 'FontSize', 12, 'FontWeight', 'bold');
% ylabel('y', 'FontSize', 12, 'FontWeight', 'bold');
% 
% legend('TRUE','RKP','ARX','DKP', 'DKISSGPR');

figure;
t = 0:0.05:5-0.1;

% 1) 先画 TRUE 并保存句柄
h_true = plot(t, true(2:end,3), '--k', 'LineWidth', lw+0.5); hold on;

% 2) 再画其它方法
h_rkp    = plot(t, kooprbf(2:end,2), 'b', 'LineWidth', lw);    hold on;
h_arx    = plot(t, arxx(2:end,1),     'g', 'LineWidth', lw);    hold on;
h_dkp    = plot(t, winer_koop',       'c', 'LineWidth', lw);    hold on;
h_dynonet =plot(t, dynonet(2:end,1), 'm', 'linewidth', lw); hold on  % Changed to magenta dashed
h_recur= plot(t,recur(2:end,1), 'Color',[1,0.5,0.2], 'linewidth', lw); hold on  % Changed to black dashed
h_dkigpr = plot(t, winer_koopg,       'r', 'LineWidth', lw+0.5);

% 3) 坐标轴、标签
xlim([0 5]);
ylim([-0.1 0.8]);
xlabel('Time(s)', 'FontSize',16,'FontWeight','bold');
ylabel('y',      'FontSize',16,'FontWeight','bold');

% 4) 按原顺序生成 legend（TRUE 在最上）
legend([h_true, h_rkp, h_arx, h_dkp,h_dynonet,h_recur, h_dkigpr], ...
       {'TRUE','RKP','ARX','DKP','dynoNet','RecReg','DKIGPR'});

% 5) 最后把 TRUE 这条线推到最上层
uistack(h_true, 'top');


errzuixiao = sqrt(mse(zuixiao(:,2),true(:,3)));
errrbf =  sqrt(mse(kooprbf(:,2),true(:,3)));
errarxx = sqrt(mse(arxx,true(:,3)));
errkoopg = sqrt(mse(winer_koopg',true(2:end,3)));
errkoop = sqrt(mse(winer_koop',true(2:end,3)));
errrecur = sqrt(mse(recur,true(1:end,3)));
errdynonet = sqrt(mse(dynonet,true(1:end,3)));

naive_forecast = true(1:end-1, 3); % 朴素预测：使用前一个时间步的值
actual_values = true(2:end, 3); % 对应的实际值

% 计算基准预测的MAE
baseline_mae = mean(abs(actual_values - naive_forecast));

% 现在计算各个模型的MASE
err_min = mase(zuixiao(:,2), true(:,3), baseline_mae);
err_rbf = mase(kooprbf(:,2), true(:,3), baseline_mae);
err_arx = mase(arxx, true(:,3), baseline_mae);
err_koop_gaussian = mase(winer_koopg', true(2:end,3), baseline_mae);
err_koop = mase(winer_koop', true(2:end,3), baseline_mae);
err_recurrent = mase(recur, true(1:end,3), baseline_mae);
err_dynonet = mase(dynonet, true(1:end,3), baseline_mae);


% 确保数据维度匹配
true_vals = true(:,3);

% 为每个模型准备对应的真实值数据
true_rbf = true_vals(1:length(kooprbf(:,2)));
true_arxx = true_vals(1:length(arxx));
true_koopg = true_vals(2:2+length(winer_koopg)-1);
true_koop = true_vals(2:2+length(winer_koop)-1);
true_recur = true_vals(1:length(recur));
true_dynonet = true_vals(1:length(dynonet));


% 计算MAPE
mape_rbf = safe_mape(true_rbf, kooprbf(:,2));
mape_arxx = safe_mape(true_arxx, arxx);
mape_koopg = safe_mape(true_koopg, winer_koopg');
mape_koop = safe_mape(true_koop, winer_koop');
mape_recur = safe_mape(true_recur, recur);
mape_dynonet = safe_mape(true_dynonet, dynonet);

% 计算MSLE
msle_rbf = mean((log1p(kooprbf(:,2)) - log1p(true_rbf)).^2);
msle_arxx = mean((log1p(arxx) - log1p(true_arxx)).^2);
msle_koopg = mean((log1p(winer_koopg') - log1p(true_koopg)).^2);
msle_koop = mean((log1p(winer_koop') - log1p(true_koop)).^2);
msle_recur = mean((log1p(recur) - log1p(true_recur)).^2);
msle_dynonet = mean((log1p(dynonet) - log1p(true_dynonet)).^2);

% 统一计算真实值的归一化因子（范围）
true_range = max(true(:,3)) - min(true(:,3)); 

% 各模型的NRMSE计算
errzuixiao1 = sqrt(mse(zuixiao(:,2), true(:,3))) / true_range;
errrbf1     = sqrt(mse(kooprbf(:,2), true(:,3))) / true_range;
errarxx1    = sqrt(mse(arxx, true(:,3))) / true_range;
errrecur1    = sqrt(mse(recur, true(:,3))) / true_range; 
errdynonet1    = sqrt(mse(dynonet, true(:,3))) / true_range; 

% 注意：以下两行若预测值长度与true(2:end,2)对应，需重新计算子集的范围
subset_true = true(2:end,3); 
subset_range = max(subset_true) - min(subset_true);

errkoopg1  = sqrt(mse(winer_koopg', subset_true)) / subset_range;
errkoop1   = sqrt(mse(winer_koop', subset_true)) / subset_range;


TEzuixiao = calculateERR(zuixiao(:,1),true(:,2));
TErbf = calculateERR(kooprbf(:,1),true(:,2));
TEarxx = calculateERR(arxx,true(:,2));
TEgauss = calculateERR(winer_koopg', true(2:end,2));
TEdp = calculateERR(winer_koop', true(2:end,2));

R2zuixiao = computeR2_multidim(zuixiao(:,2),true(:,3));
R2rbf = computeR2_multidim(kooprbf(:,2),true(:,3));
R2arxx = computeR2_multidim(arxx,true(:,3));
R2gauss = computeR2_multidim(winer_koopg', true(2:end,3));
R2recur = computeR2_multidim(recur, true(1:end,3));
R2dgynonet = computeR2_multidim(dynonet, true(1:end,3));


function result = ARXEst(u,y,u2,y2,order,result)
umin=min(u2);umax=max(y2);
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


% 安全MAPE计算函数（避免除零错误）
function mape_val = safe_mape(true_data, pred_data)
    valid_idx = true_data ~= 0 & ~isnan(true_data) & ~isnan(pred_data);
    if sum(valid_idx) == 0
        mape_val = NaN;
    else
        mape_val = mean(abs((true_data(valid_idx) - pred_data(valid_idx)) ./ ...
                      true_data(valid_idx))) * 100;
    end
end

% 需要先定义MASE函数（如果MATLAB中没有内置的话）
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

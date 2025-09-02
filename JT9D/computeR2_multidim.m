function R2 = computeR2_multidim( y_pred,y_true)
    % 计算 R^2 (决定系数) 适用于多维输出
    % 输入:
    % y_true - 实际值 (N×2 矩阵)
    % y_pred - 预测值 (N×2 矩阵)
    %
    % 输出:
    % R2 - 每个维度的 R^2 值 (1×2 向量)

    % 检查输入维度
    if ~isequal(size(y_true), size(y_pred))
        error('y_true 和 y_pred 的尺寸必须相同');
    end

    % 获取维度数
    [N, dim] = size(y_true);
    
    % 初始化 R^2
    R2 = zeros(1, dim);

    % 对每列（每个维度）计算 R^2
    for d = 1:dim
        y_true_col = y_true(:, d); % 第 d 列的实际值
        y_pred_col = y_pred(:, d); % 第 d 列的预测值
        
        % 平均值
        y_mean = mean(y_true_col);

        % 总平方和 (Total Sum of Squares)
        SS_tot = sum((y_true_col - y_mean).^2);

        % 残差平方和 (Residual Sum of Squares)
        SS_res = sum((y_true_col - y_pred_col).^2);

        % 计算 R^2
        R2(d) = 1 - (SS_res / SS_tot);
    end
end

#! Full Replication
# python main.py \
# --data_path ../NCSOFT/financial_data \
# --result_path ./results \
# --solution_name lagrange_full \
# --start_date "2018-01-02" \
# --end_date "2018-12-31" \
# --backtesting False \
# --month_increment 3 \
# --index_type "s&p500" 
# --method SLSQP


# python main.py \
# --data_path ../NCSOFT/financial_data \
# --result_path ./results \
# --solution_name QP_full \
# --start_date "2018-01-02" \
# --end_date "2018-12-31" \
# --backtesting False \
# --month_increment 3 \
# --index_type "s&p500" 



#! Partial Replication
# # 이거!!!!
python main_lag.py \
--data_path ../NCSOFT/financial_data \
--result_path ./results \
--solution_name lagrange_ours \
--cardinality 50 \
--start_date "2018-01-02" \
--end_date "2023-04-28" \
--backtesting True \
--month_increment 3 \
--index_type "kospi100" \
--method SLSQP

python main_lag.py \
--data_path ../NCSOFT/financial_data \
--result_path ./results \
--solution_name lagrange_forward \
--cardinality 50 \
--start_date "2018-01-02" \
--end_date "2023-04-28" \
--backtesting True \
--month_increment 3 \
--index_type "kospi100" \
--method SLSQP

python main_lag.py \
--data_path ../NCSOFT/financial_data \
--result_path ./results \
--solution_name lagrange_backward \
--cardinality 50 \
--start_date "2018-01-02" \
--end_date "2023-04-28" \
--backtesting True \
--month_increment 3 \
--index_type "kospi100" \
--method SLSQP

python main_lag.py \
--data_path ../NCSOFT/financial_data \
--result_path ./results \
--solution_name QP_forward \
--cardinality 50 \
--start_date "2018-01-02" \
--end_date "2023-04-28" \
--backtesting True \
--month_increment 3 \
--index_type "kospi100"
# --method SLSQP

python main_lag.py \
--data_path ../NCSOFT/financial_data \
--result_path ./results \
--solution_name QP_backward \
--cardinality 50 \
--start_date "2018-01-02" \
--end_date "2023-04-28" \
--backtesting True \
--month_increment 3 \
--index_type "kospi100"
# --method SLSQP

python main_lag.py \
--data_path ../NCSOFT/financial_data \
--result_path ./results \
--solution_name SNN \
--cardinality 50 \
--start_date "2018-01-02" \
--end_date "2023-04-28" \
--backtesting True \
--month_increment 3 \
--index_type "kospi100" 

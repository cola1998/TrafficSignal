# -*- coding:utf-8 -*-
import xlrd
import xlsxwriter

def record_data(file_name,d):

    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    name_list = list(d.keys())
    for j in range(len(name_list)):
        worksheet.write(0, j, name_list[j])

    for i in range(len(d[name_list[0]])):  # i 代表行 j代表列
        for j in range(len(name_list)):
            worksheet.write(i+1,j,d[name_list[j]][i])

    workbook.close()

def record_data_ft(ft_waiting_time_list,ft_delay_time_list,ft_queue_list,ft_remain_car_numbers,file_name):
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    name_list = ['waiting_time', 'delay_time', 'queue', 'throughput']
    for j in range(len(name_list)):
        worksheet.write(0, j, name_list[j])
    k = len(name_list) - 1
    for i in range(len(ft_waiting_time_list)):
        j = 0
        worksheet.write(i + 1, j, ft_waiting_time_list[i])
        j += 1
        worksheet.write(i + 1, j, ft_delay_time_list[i])
        j += 1
        worksheet.write(i + 1, j, ft_queue_list[i])
        j += 1
        worksheet.write(i + 1, j, ft_remain_car_numbers[i])

    workbook.close()
#
# d = {'loss':[1,2,3],'reward':[4,5,6]}
# record_data("test.xlsx",d)
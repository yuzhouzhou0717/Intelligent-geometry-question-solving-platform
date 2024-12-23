# !/usr/bin/python
# -*- coding: utf-8 -*--
# 讲结果本地输出，方便调试


from wolframclient.evaluation import WolframLanguageSession
import os
import cv2 as cv
import time
import pandas as pd
import numpy as np
from assemble import inverse_semantics, find_same_formula, exercise_output
from interact_with_GPT import analysis_exercise

from point_slove import FindIdentity
from Output import Mapping, genereate_title, generate_answer
from Generate import Generate
import pickle

PrintIntermediate = True  # 是否输出中间结果

if __name__ == '__main__':
    # print('开始时间:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\n')
    with WolframLanguageSession() as WL_session:

        # 1.题意理解
        # seed_list = ['如图，由△ABC的两边向外做正方形BADE和BCGF，P是平面内一点，满足PG⊥AB、PD⊥CB. 求证：PB⊥AC.',
        #              '如图，△ABC中，AB=AC，D在BC上，E在AB上且DB=DE. 求证：A、E、D、C四点共圆.']
        #
        seed_list = ['如图，三角形ABC中，∠BAC=90°，M、F、E分别为BC、CA、AB 的中点。求证:EF=AM.',
                     '如图，梯形ABCD中，AD//BC，∠A=90°，E为AB中点，DE⊥CE.求证：AB^2=4*AD*BC.'] #种子习题题干

        exercise_list=[] #种子习题题干解析结果，用谓词语句表示
        for seed in seed_list:
            result = analysis_exercise(seed)
            if result[0] == 0:
                exercise_list.append(result[1])
            else:
                exit('题干解析出错！')

        # 2. 点几何恒等式求解，几何关系提取
        expression_list = []
        for i in range(0, len(exercise_list)):
            exercise = exercise_list[i]
            find_result = FindIdentity(exercise['已知条件'], exercise['待求结论'], WL_session)
            if find_result[0] == 0:
                for jj in range(len(find_result[2])):
                    one_expression = find_result[2][jj]
                    expression_list.append([i] + one_expression + [jj])

        expression_list = pd.DataFrame(expression_list,
                                     columns=['Eid', 'Point', 'Expression', 'Predicate', 'Eliminate', 'Draw',
                                              'NO'])  # 习题编号，点，表达式，谓词，替换关系，画图语句，编号

        expression_list.to_excel(r'./expression_list.xlsx', index=False) #子式提取几何关系提取结果

        with open(os.path.join(os.path.dirname(__file__), r'expression_list.pkl'), 'wb') as f:
            pickle.dump(expression_list, f)

        with open(os.path.join(os.path.dirname(__file__), r'expression_list.pkl'), 'rb') as f:
            expression_list = pickle.load(f)

        # 3.获取子式的逆语义
        inverse_result = inverse_semantics(expression_list)
        if inverse_result[0] == 0:
            expression_list_with_inverse = inverse_result[1]
            expression_list_with_inverse.to_excel(r'./expression_list_with_inverse.xlsx', index=False)  # 子式有逆语义
        else:
            exit('获取逆语义出错！')


        # 4.通过检测同构式子是否想等，形成可能的新习题
        find_same_result = find_same_formula(expression_list_with_inverse)
        if find_same_result[0] == 0:
            new_exercise = find_same_result[1]
            new_exercise.to_excel(r'./new_exercise.xlsx', index=False)  # 子式提取几何关系提取结果
        else:
            exit('表达式同构检测出错')

        # 5.新习题输出
        out_put_result=exercise_output(new_exercise)
        if out_put_result[0]==0:
            out_put_result[1].to_excel(r'./out_result.xlsx', index=False)
        else:
            exit('习题输出出错')


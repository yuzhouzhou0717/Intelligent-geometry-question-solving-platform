import sys
import os
import django
from django.http import HttpResponse
from django.shortcuts import render
from wolframclient.evaluation import WolframLanguageSession
from products.assemble import inverse_semantics, find_same_formula, exercise_output
from products.interact_with_GPT import analysis_exercise
from products.point_slove import FindIdentity
import pandas as pd
import pickle
from products.models import Cart
from django.conf import settings
# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject2.settings')
django.setup()



def execute_code(request):
    PrintIntermediate = True  # 是否输出中间结果
    # 设置 Django 环境
    # os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject2.settings')
    #  django.setup()

    with WolframLanguageSession() as WL_session:
        exercise_list = []
        cart_items = Cart.objects.all()
        for item in cart_items:
            result = analysis_exercise(item.stem)
            print(result)
            if result[0] == 0:
                exercise_list.append(result[1])
            else:
                print('题干解析出错！')

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

        expression_list.to_excel(r'./expression_list.xlsx', index=False)

        with open(os.path.join(os.path.dirname(__file__), r'expression_list.pkl'), 'wb') as f:
            pickle.dump(expression_list, f)

        with open(os.path.join(os.path.dirname(__file__), r'expression_list.pkl'), 'rb') as f:
            expression_list = pickle.load(f)

        inverse_result = inverse_semantics(expression_list)
        if inverse_result[0] == 0:
            expression_list_with_inverse = inverse_result[1]
            expression_list_with_inverse.to_excel(r'./expression_list_with_inverse.xlsx', index=False)
        else:
            print('获取逆语义出错！')

        find_same_result = find_same_formula(expression_list_with_inverse)
        if find_same_result[0] == 0:
            new_exercise = find_same_result[1]
            new_exercise.to_excel(r'./new_exercise.xlsx', index=False)
        else:
            print('表达式同构检测出错')

        out_put_result = exercise_output(new_exercise)
        if out_put_result[0] == 0:
            out_put_result[1].to_excel(r'./out_result.xlsx', index=False)
        else:
            print('习题输出出错')

    print('代码执行结束')
    return HttpResponse("Code executed successfully")

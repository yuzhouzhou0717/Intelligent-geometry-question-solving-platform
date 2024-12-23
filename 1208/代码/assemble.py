from sympy import symbols, sympify, Abs, nonlinsolve, sqrt, Mul, simplify, solveset, linsolve, latex, expand, dotprint
from re import match, compile, sub
import re
import time
import numpy as np
import cv2 as cv
import math
from sympy.abc import _clash1, _clash2, _clash  # 类似 E、I　等冲突字母处理　
import pickle
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr
from wolframclient.language.expression import WLFunction, WLSymbol
import random
from itertools import permutations, combinations
import copy
from func_timeout import func_set_timeout, exceptions
import shutil
import os
import pandas as pd
from collections import Counter
from judge_convex_hull import convex_hull
from random import shuffle
import torch
from sympy import Matrix

'''
基于已有习题合成新习题。
几何关系提取的时候要解出习题，找打它的逆含义
组装可行性通过符号计算实现
习题组装的最原始方法
'''


@func_set_timeout(10)  # 函数执行时间为6秒
def solve_equations_mathematica(wolframc_str: str, session: WolframLanguageSession):
    '''
    # 调用mathematica 解方程和不等式组
   :param wolframc_str: mathematica 语句表示的方程和不等式组
    :param session: 调用mathematica 会话
    :return: 结果
    '''
    return session.evaluate(wlexpr(wolframc_str))


def solve_equations(wolframc_str: str, session: WolframLanguageSession):
    '''
    调用mathematica 解等式和不等式系统，求解

    :param wolframc_str: mathematica 语句表示的方程和不等式组
    :param session: 调用mathematica 会话
    :return: （id,value）
        id<0  计算出现异常
        id==0 约束系统有解
        id==1 约束系统无解
    '''
    # print(wolframc_str)

    # 1.字符串中一些字符替换
    # if ('sqrt' in wolframc_str):
    #     return (-1, '方程中有错误字符___sqrt 出现')
    wolframc_str = wolframc_str.replace('sqrt', 'Sqrt')  # 根号替换
    wolframc_str = wolframc_str.replace('**', '^')  # 幂运算替换

    # 2.解约束系统
    try:
        result = solve_equations_mathematica(wolframc_str, session)
    except exceptions.FunctionTimedOut:
        session.terminate()
        return (-2, '解约束系统超时')
    except Exception as e:
        return (-1, '解约束系统出错___' + str(e))

    if ('Global' in str(result)):
        return (1, '约束系统无解')

    return (0, '约束系统有解', result)


def get_circle(x1, y1, x2, y2, x3, y3):
    """
    三点共圆，计算圆心和半径
    :return:  x0 and y0 is center of a circle, r is radius of a circle
    """
    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3
    a1 = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / sympify(2)
    a2 = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / sympify(2)
    theta = b * c - a * d
    if abs(theta) < 1e-7:
        raise RuntimeError('There should be three different x & y !')
    x0 = (b * a2 - d * a1) / theta
    y0 = (c * a1 - a * a2) / theta
    r = float(sqrt(pow((x1 - x0), 2) + pow((y1 - y0), 2)))
    return int(round(x0)), int(round(y0)), int(round(r))


def get_circle_symbolic(x1, y1, x2, y2, x3, y3):
    """
    三点共圆，计算圆心和半径，符号计算，返回精确值
    :return:  x0 and y0 is center of a circle
    """
    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3
    a1 = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / sympify(2)
    a2 = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / sympify(2)
    theta = b * c - a * d
    x0 = (b * a2 - d * a1) / theta
    y0 = (c * a1 - a * a2) / theta
    return (x0, y0)


def construc_edge(edge_dic: dict, new_edge_list: list):
    '''
    构建几何图形的连边信息，存储在edge_dic 中

    :param edge_dic:几何图形的连边信息
    :param new_edge_list: 新输入的边
    :return:
    '''
    # print(new_edge_list)
    for edge in new_edge_list:
        p1, p2 = edge

        if p1 not in edge_dic.keys():
            edge_dic[p1] = {p2: 1}  # 这里由问题，不应该统计次数，而是统计连边由哪些节点
        elif p2 not in edge_dic[p1].keys():
            edge_dic[p1][p2] = 1
        else:
            edge_dic[p1][p2] += 1

        if p2 not in edge_dic.keys():
            edge_dic[p2] = {p1: 1}
        elif p1 not in edge_dic[p2].keys():
            edge_dic[p2][p1] = 1
        else:
            edge_dic[p2][p1] += 1


def draw_geometric_element(img: np.ndarray, edge_dict: dict, point_draw: dict, colour: tuple, sematic: str, thickness=2,
                           lineType=cv.LINE_AA):
    '''
     根据题干语句，画线段、多边形、圆等几何元素
    :param img: 图形数据
    :param edge_dict:  边的信息，用于找孤立点
    :param point_draw:  点坐标
    :param colour:  颜色
    :param sematic: 题干语句
    :return: img和edge_dict形参调用；sematic主动返回
    '''

    # 画线段
    for element in set(compile(r'([A-Z][A-Z])').findall(sematic)):  # 两个字母连载一起的只有线段
        result = compile(r'([A-Z])').findall(element)
        p1, p2 = result  # 节点名称
        pt1, pt2 = point_draw[p1], point_draw[p2]
        cv.line(img, pt1, pt2, colour, thickness, lineType)
        construc_edge(edge_dict, [(p1, p2)])
        # 形式语言需要修改，线段名称统一从左到右，从上到下
        if pt1[0] < pt2[0]:
            pass
        elif pt1[0] == pt2[0] and pt1[1] > pt2[1]:
            pass
        else:
            sematic = sematic.replace('{}{}'.format(p1, p2), '{}{}'.format(p2, p1))

    ##画延长线   line_intersection_at(AE,CF,D)
    for element in set(
            compile(r'line_intersection_at\([A-Z][A-Z],[A-Z][A-Z],[A-Z]\)').findall(sematic)):
        p1, p2, p3 = compile(r'line_intersection_at\([A-Z]([A-Z]),[A-Z]([A-Z]),([A-Z])\)').search(element).groups()

        pt1 = point_draw[p1]
        pt2 = point_draw[p2]
        pt3 = point_draw[p3]

        cv.line(img, pt1, pt3, colour, thickness, lineType)
        construc_edge(edge_dict, [(p1, p3)])
        cv.line(img, pt2, pt3, colour, thickness, lineType)
        construc_edge(edge_dict, [(p2, p3)])

    # 画直角三角形斜边高
    # for tu in set(compile(r'RightTriangle\([A-Z],[A-Z],[A-Z]\),Line\([A-Z],[A-Z]\),Pedal\([A-Z]\)').findall(hypothesis)):
    #     if tu in conclusion_sematic:
    #         colour = red
    #     else:
    #         colour = black
    #     result = compile(r'RightTriangle\([A-Z],([A-Z]),([A-Z])\),Line\([A-Z],[A-Z]\),Pedal\(([A-Z])\)').match(tu)
    #     p1, p2, p3 = result.groups()  # 名称
    #
    #     pt1 = point_draw[p1]
    #     pt3 = point_draw[p3]
    #
    #     cv.line(img, pt1, pt3, colour, thickness, lineType)
    #     construc_edge(edge_dict, [(p1, p3)])
    #     construc_edge(edge_dict, [(p2, p3)])

    # 其他几何元素
    for sentence in sematic.split(';'):  # 就这样群分开地画，在修改形式和语言的时候，容易控制，现在不要修改，后面再修改。
        for entity in sentence.split('__'):  # 两个下划线拼接不同的语句，一个下划线链接单词
            if entity == '':
                continue
            index = entity.find('(')
            if (index == -1):
                raise RuntimeError('画几何实体时出错————{}'.format(entity))
            key = entity[0:index]

            if key == 'collinear':
                result = compile(r'collinear\(([A-Z]),([A-Z]),([A-Z])\)').match(entity)

                p1, p2, p3 = result.groups()  # 名称
                pt_dic = {}
                for p in (p1, p2, p3):
                    pt_dic[p] = point_draw[p]
                sorted_result = sorted(Counter(pt_dic).items(), key=lambda item: item[1], reverse=False)
                p1, p2, p3 = sorted_result[0][0], sorted_result[1][0], sorted_result[2][0]
                sematic = sematic.replace(result.string,
                                          'collinear({},{},{})'.format(p1, p2, p3))

                pt1, pt3 = point_draw[p1], point_draw[p3]
                cv.line(img, pt1, pt3, colour, thickness, lineType)

            elif key == 'angle':
                result = compile(r'angle\(([A-Z]),([A-Z]),([A-Z])\)').match(entity)
                p1, p2, p3 = result.groups()  # 名称

                pt1 = point_draw[p1]
                pt2 = point_draw[p2]
                pt3 = point_draw[p3]

                if pt1[0] > pt3[0] or pt1[0] == pt3[0] and pt1[1] > pt3[1]:  # 几何实体名称统一从左到右，从上到下
                    sematic = sematic.replace(result.string, 'angle({},{},{})'.format(p3, p2, p1))

                pts = np.array([pt1, pt2, pt3])
                pts = pts.reshape((-1, 1, 2))
                cv.polylines(img, [pts], False, colour, thickness, lineType)
                construc_edge(edge_dict, [(p1, p2), (p2, p3)])

            elif key == 'angle_equal':  # angle_equal(AON) 角度值；特殊角度值  60度 120度
                result = compile(r'angle_equal\(([A-Z]),([A-Z]),([A-Z]),([0-9/-]*)\)').match(entity)
                p1, p2, p3, value = result.groups()
                pt1 = point_draw[p1]
                pt2 = point_draw[p2]
                pt3 = point_draw[p3]

                if pt1[0] > pt3[0] or pt1[0] == pt3[0] and pt1[1] > pt3[1]:  # 几何实体名称统一从左到右，从上到下
                    sematic = sematic.replace(result.string, 'angle_equal({},{},{},{})'.format(p3, p2, p1, value))

                pts = np.array([pt1, pt2, pt3])
                pts = pts.reshape((-1, 1, 2))
                cv.polylines(img, [pts], False, colour, thickness, lineType)
                construc_edge(edge_dict, [(p1, p2), (p2, p3)])

            # symmetric(H,BC,E)
            elif (key == 'symmetric'):
                result = compile(r'symmetric\(([A-Z]),[A-Z][A-Z],([A-Z])\)').match(entity)
                p1, p2 = result.groups()  # 节点名称
                pt1 = point_draw[p1]
                pt2 = point_draw[p2]
                cv.line(img, pt1, pt2, colour, thickness, lineType)
                construc_edge(edge_dict, [(p1, p2)])

            elif key in ['triangle', 'regular_triangle', 'right_triangle']:  # 三角形
                if key == 'triangle':
                    result = compile(r'triangle\(([A-Z]),([A-Z]),([A-Z])\)').match(entity)
                elif key == 'regular_triangle':
                    result = compile(r'regular_triangle\(([A-Z]),([A-Z]),([A-Z])\)').match(entity)  # 正三角形
                elif key == 'right_triangle':
                    result = compile(r'right_triangle\(([A-Z]),([A-Z]),([A-Z])\)').match(entity)  # 直角三角形

                p1, p2, p3 = result.groups()  # 点名称
                pt_dic = {}
                for p in result.groups():
                    pt_dic[point_draw[p]] = p
                pts_coo = list(pt_dic.keys())  # 坐标
                pts_coo_sort = convex_hull(pts_coo)  # 点逆时针排序后的坐标

                if (str(pts_coo) != str(pts_coo_sort)):  # 形式语言需要修改
                    p1, p2, p3 = [pt_dic[pt] for pt in pts_coo_sort]
                    sematic = sematic.replace(result.string, '{}({},{},{})'.format(key, p1, p2, p3))
                    pts_coo = pts_coo_sort  # 画图时的坐标顺序 也修改

                pts = np.array(pts_coo)
                pts = pts.reshape((-1, 1, 2))
                cv.polylines(img, [pts], True, colour, thickness, lineType)
                construc_edge(edge_dict, [(p1, p2), (p2, p3), (p3, p1)])

            elif key in ('quadrilateral', 'square', 'parallelogram', 'trapezoid'):  # 四边形
                if key == 'quadrilateral':
                    result = compile(r'quadrilateral\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').match(entity)
                elif key == 'square':
                    result = compile(r'square\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').match(entity)
                elif key == 'parallelogram':
                    result = compile(r'parallelogram\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').match(entity)
                elif key == 'trapezoid':
                    result = compile(r'trapezoid\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').match(entity)

                p1, p2, p3, p4 = result.groups()  # 名称
                pt_dic = {}
                for p in result.groups():
                    pt_dic[point_draw[p]] = p
                pts_coo = list(pt_dic.keys())  # 坐标
                pts_coo_sort = convex_hull(pts_coo)  # 点逆时针排序

                if (str(pts_coo) != str(pts_coo_sort)):  # 形式语言需要修改
                    p1, p2, p3, p4 = [pt_dic[pt] for pt in pts_coo_sort]  # 有是返回3个点？
                    sematic = sematic.replace(result.string, '{}({},{},{},{})'.format(key, p1, p2, p3, p4))
                    pts_coo = pts_coo_sort  # 画图时的坐标顺序 也修改

                pts = np.array(pts_coo)
                pts = pts.reshape((-1, 1, 2))
                cv.polylines(img, [pts], True, colour, thickness, lineType)
                construc_edge(edge_dict, [(p1, p2), (p2, p3), (p3, p4), (p4, p1)])

            elif (key == '4_points_on_circle'):  # 圆
                result = compile(r'4_points_on_circle\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').match(entity)
                p1, p2, p3, p4 = result.groups()  # 名称
                pt1 = point_draw[p1]
                pt2 = point_draw[p2]
                pt3 = point_draw[p3]
                c1, c2, r = get_circle(pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1])
                cv.circle(img, (c1, c2), r, colour, thickness, lineType)  # 目标图像，圆心，半径，颜色，线宽，线型

                pt_dic = {}  # 坐标：名称
                for p in result.groups():
                    pt_dic[point_draw[p]] = p
                pts_coo = list(pt_dic.keys())  # 坐标
                pts_coo_sort = convex_hull(pts_coo)  # 点逆时针排序后的坐标

                if (str(pts_coo) != str(pts_coo_sort)):  # 形式语言需要修改
                    p1, p2, p3, p4 = [pt_dic[pt] for pt in pts_coo_sort]
                    sematic = sematic.replace(result.string, '4_points_on_circle({},{},{},{})'.format(p1, p2, p3, p4))

            elif key == 'cyclic_quadrilateral':  # 四边形外接圆
                result = compile(
                    'cyclic_quadrilateral\(([A-Z]),quadrilateral\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)\)').match(
                    entity)
                p1, p2, p3, p4, p5 = result.groups()  # 名称
                pt1 = point_draw[p1]
                pt2 = point_draw[p2]
                r = int(round(np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)))
                cv.circle(img, pt1, r, colour, thickness, lineType)  # 目标图像，圆心，半径，颜色，线宽，线型

                pt_dic = {}
                for p in (p2, p3, p4, p5):
                    pt_dic[point_draw[p]] = p
                pts_coo = list(pt_dic.keys())  # 坐标
                pts_coo_sort = convex_hull(pts_coo)  # 点逆时针排序

                if (str(pts_coo) != str(pts_coo_sort)):  # 形式语言需要修改
                    p2, p3, p4, p5 = [pt_dic[pt] for pt in pts_coo_sort]  # 有是返回3个点？
                    sematic = sematic.replace(result.string, 'quadrilateral({},{},{},{})'.format(p2, p3, p4, p5))
                    pts_coo = pts_coo_sort  # 画图时的坐标顺序 也修改

                pts = np.array(pts_coo)
                pts = pts.reshape((-1, 1, 2))
                cv.polylines(img, [pts], True, colour, thickness, lineType)
                construc_edge(edge_dict, [(p2, p3), (p3, p4), (p4, p5), (p5, p1)])


            elif key in ['circumcenter']:  # circumcenter(H,triangle(A,B,C)) 三角形外的外接外心
                result = compile('circumcenter\(([A-Z]),triangle\(([A-Z]),([A-Z]),([A-Z])\)\)').match(
                    entity)
                p1, p2, p3, p4 = result.groups()  # 名称
                pt1 = point_draw[p1]
                pt2 = point_draw[p2]
                r = int(round(np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)))
                cv.circle(img, pt1, r, colour, thickness, lineType)  # 目标图像，圆心，半径，颜色，线宽，线型

                pt_dic = {}
                for p in (p2, p3, p4):
                    pt_dic[point_draw[p]] = p
                pts_coo = list(pt_dic.keys())  # 坐标
                pts_coo_sort = convex_hull(pts_coo)  # 点逆时针排序后的坐标

                if (str(pts_coo) != str(pts_coo_sort)):  # 形式语言需要修改
                    n_p2, n_p3, n_p4 = [pt_dic[pt] for pt in pts_coo_sort]
                    sematic = sematic.replace('triangle({},{},{})'.format(p2, p3, p4),
                                              'triangle({},{},{})'.format(n_p2, n_p3, n_p4))
                    pts_coo = pts_coo_sort  # 画图时的坐标顺序 也修改

                pts = np.array(pts_coo)
                pts = pts.reshape((-1, 1, 2))
                cv.polylines(img, [pts], True, colour, thickness, lineType)
                construc_edge(edge_dict, [(p1, p2), (p2, p3), (p3, p1)])

            elif key in ['barycenter']:  # barycenter(H,triangle(A,B,C)) 三角形外的外接重心
                result = compile('barycenter\(([A-Z]),triangle\(([A-Z]),([A-Z]),([A-Z])\)\)').match(
                    entity)
                p1, p2, p3, p4 = result.groups()  # 名称

                pt_dic = {}
                for p in (p2, p3, p4):
                    pt_dic[point_draw[p]] = p
                pts_coo = list(pt_dic.keys())  # 坐标
                pts_coo_sort = convex_hull(pts_coo)  # 点逆时针排序后的坐标

                if (str(pts_coo) != str(pts_coo_sort)):  # 形式语言需要修改
                    n_p2, n_p3, n_p4 = [pt_dic[pt] for pt in pts_coo_sort]
                    sematic = sematic.replace('triangle({},{},{})'.format(p2, p3, p4),
                                              'triangle({},{},{})'.format(n_p2, n_p3, n_p4))
                    pts_coo = pts_coo_sort  # 画图时的坐标顺序 也修改

                pts = np.array(pts_coo)
                pts = pts.reshape((-1, 1, 2))
                cv.polylines(img, [pts], True, colour, thickness, lineType)
                construc_edge(edge_dict, [(p1, p2), (p2, p3), (p3, p1)])

            elif (key == 'circle_from_3_points'):  # 三点画圆 circle_from_3_points(A,B,C)
                result = compile('circle_from_3_points\(([A-Z]),([A-Z]),([A-Z])\)').match(
                    entity)
                p1, p2, p3 = result.groups()
                pt1 = point_draw[p1]
                pt2 = point_draw[p2]
                pt3 = point_draw[p3]
                c1, c2, r = get_circle(pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1])
                cv.circle(img, (c1, c2), r, colour, thickness, lineType)  # 目标图像，圆心，半径，颜色，线宽，线型

            elif (key == 'point_on_circle_from_diameter'):  # 直径画圆 point_on_circle_from_diameter(D,BC)
                result = compile('point_on_circle_from_diameter\(([A-Z]),([A-Z])([A-Z])\)').match(
                    entity)
                p1, p2, p3 = result.groups()
                pt2 = point_draw[p2]
                pt3 = point_draw[p3]

                c1, c2 = (pt2[0] + pt3[0]) / 2, (pt2[1] + pt3[1]) / 2  # 圆心坐标
                r = int(round(np.sqrt((c1 - pt2[0]) ** 2 + (c2 - pt2[1]) ** 2)))
                c1, c2 = int(round(c1)), int(round(c2))
                cv.circle(img, (c1, c2), r, colour, thickness, lineType)  # 目标图像，圆心，半径，颜色，线宽，线型
            else:
                pass
                # print('有未识别的画图语句：', entity)

    return sematic


def genereate_title(paragraphs: str):
    '''
    生成自然语言形式的题干
    知识点排序：  1平行四边形(1.1矩形 1.2菱形  1.3正方向)  2梯形  3四边形  4圆  5三角形  6线段共线(6.1共线   6.2一个中点， 6.3中点融合)  7平行  8垂直   9数量关系, 10 其他
    底层5个知识点，顶层
    :param paragraph: 几何形式语言
    :return:
        ralation_list：生成解答时用
        title：形成的自然语言题干
    '''

    sentence_dic = {}  # 形式语言去重
    paragraph_sort = []  # 排序后的，题目中不同知识点的出现次序应该要固定

    for paragraph in paragraphs.split(';'):
        one_relation = []
        for sentence in paragraph.split('__'):
            if sentence.replace(' ', '') == '':
                continue
            if sentence in sentence_dic.keys():  # 形式语言去重，也是语句去重
                continue
            else:
                sentence_dic[sentence] = 1

            index = sentence.find('(')
            if (index == -1):
                return (-1, '几何形式语言有错^^^^^!')

            if index == 0:  # 直接是自然语言
                one_relation.append((sentence[1:-1], -1))
                continue

            key = sentence[0:index]
            if key == 'free_point':
                continue
                pattern = compile(
                    r'free_point\(([A-Z])\)')
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 0))
                    continue

                p1 = result.groups()
                title = '点'.format(p1)
                one_relation.append((title, 0))

            elif key == 'collinear':
                # collinear(C,M,D)__equal(CM,2/3*MD)
                pattern = compile(
                    r'collinear\(([A-Z]),([A-Z]),([A-Z])\)')
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 1))
                    continue
                p1, p2, p3 = result.groups()
                title = '{0}、{1}、{2}共线'.format(p1, p2, p3)
                one_relation.append((title, 1))

            elif key in ['midpoint', 'lieson']:
                # middle(D,CB)
                pattern = compile(
                    r'{}\(([A-Z]),([A-Z])([A-Z])\)'.format(key))
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 2))
                    continue
                p1, p2, p3 = result.groups()
                if key == 'midpoint':
                    title = '{}是{}{}的中点'.format(p1, p2, p3)
                elif key == 'lieson':
                    title = '{}在线段{}{}上'.format(p1, p2, p3)
                one_relation.append((title, 2))

            elif key == 'line_intersection_at':
                # line_intersection_at(BD,CE,A)
                pattern = compile(
                    r'line_intersection_at\(([A-Z])([A-Z]),([A-Z])([A-Z]),([A-Z])\)')
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 2.1))
                p1, p2, p3, p4, p5 = result.groups()
                title = '{}{}、{}{}交于点{}'.format(p1, p2, p3, p4, p5)

                one_relation.append((title, 2.1))

            elif key in ['parallel', 'perpendicular', 'line_less']:
                pattern = compile(
                    r'{}\(([A-Z])([A-Z]),([A-Z])([A-Z])\)'.format(key))
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 3))
                    continue

                p1, p2, p3, p4 = result.groups()
                if key == 'parallel':
                    title = '{}{}//{}{}'.format(p1, p2, p3, p4)
                if key == 'perpendicular':
                    title = '{}{}⊥{}{}'.format(p1, p2, p3, p4)
                if key == 'line_less':
                    title = '{}{}小于{}{}'.format(p1, p2, p3, p4)
                one_relation.append((title, 3))

            elif key == 'same_side':  # same_side(G,D,AB)
                pattern = compile(
                    r'same_side\(([A-Z]),([A-Z]),([A-Z])([A-Z])\)')
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 4))
                    continue

                p1, p2, p3, p4 = result.groups()
                title = '{}、{}在直线{}{}的同侧'.format(p1, p2, p3, p4)
                one_relation.append((title, 4))
            elif key == 'both_side':  # both_side(G,D,AB)
                pattern = compile(
                    r'both_side\(([A-Z]),([A-Z]),([A-Z])([A-Z])\)')
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 4))
                    continue

                p1, p2, p3, p4 = result.groups()
                title = '{}、{}在直线{}{}的两侧'.format(p1, p2, p3, p4)
                one_relation.append((title, 4))

            elif key == 'symmetric':
                pattern = compile(
                    r'symmetric\(([A-Z]),([A-Z])([A-Z]),([A-Z])\)')
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 4))
                    continue

                p1, p2, p3, p4 = result.groups()
                title = '{}、{}关于{}{}对称'.format(p1, p4, p2, p3)
                one_relation.append((title, 4))

            elif key == 'angle_equal':
                # angle_equal(A,C,B,1/2)
                pattern = compile(
                    r'angle_equal\(([A-Z]),([A-Z]),([A-Z]),([0-9/-]*)\)')
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 5))
                    continue
                p1, p2, p3, value = result.groups()
                value = sympify(value)
                if value == 1 / 2:
                    title = '角{}{}{}=60度'.format(p1, p2, p3, value)
                elif value == -1 / 2:
                    title = '角{}{}{}=120度'.format(p1, p2, p3, value)

                one_relation.append((title, 5))

            elif key in ['triangle', 'regular_triangle']:
                # 'triangle(B,D,A)
                pattern = compile(
                    r'{}\(([A-Z]),([A-Z]),([A-Z])\)'.format(key))
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 6))
                    continue

                p1, p2, p3 = result.groups()
                if key == 'triangle':
                    title = '三角形{}{}{}'.format(p1, p2, p3)
                    one_relation.append((title, 6))
                if key == 'regular_triangle':
                    title = '正三角形{}{}{}'.format(p1, p2, p3)
                    one_relation.append((title, 6))

            elif key in ['orthocenter', 'circumcenter', 'barycenter']:
                # orthocenter(P,triangle(A,B,C))
                pattern = compile(
                    r'{}\(([A-Z]),triangle\(([A-Z]),([A-Z]),([A-Z])\)\)'.format(key))
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 6))
                    continue
                p1, p2, p3, p4 = result.groups()
                if key == 'orthocenter':
                    title = '{}是三角形{}{}{}的垂心'.format(p1, p2, p3, p4)
                elif key == 'circumcenter':
                    title = '{}是三角形{}{}{}的外心'.format(p1, p2, p3, p4)
                elif key == 'barycenter':
                    title = '{}是三角形{}{}{}的重心'.format(p1, p2, p3, p4)
                one_relation.append((title, 6))

            elif key in ('parallelogram', 'quadrilateral', 'square', 'trapezoid'):
                # 'parallelogram(A,D,C,E)'
                pattern = compile(r'{}\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)'.format(key))
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 7))
                p1, p2, p3, p4 = result.groups()

                if key == 'parallelogram':
                    title = '平行四边形{}{}{}{}'.format(p1, p4, p3, p2)
                elif key == 'quadrilateral':
                    title = '四边形{}{}{}{}'.format(p1, p4, p3, p2)
                elif key == 'square':
                    title = '正方形{}{}{}{}'.format(p1, p4, p3, p2)
                elif key == 'trapezoid':
                    title = '梯形{}{}{}{}'.format(p1, p4, p3, p2)
                one_relation.append((title, 7))

            # elif key == 'trapezoid':
            #     # trapezoid(A,B,C,D)__parallel(AD,BC)__equal(AD,3*BC)
            #     pattern = compile(
            #         r'trapezoid\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)__parallel\([A-Z][A-Z],[A-Z][A-Z]\)__equal\(([A-Z])([A-Z]),([0-9/-]*)\*([A-Z])([A-Z])\)')
            #     result = pattern.match(sentence)
            #     if (result == None):
            #         one_relation.append((sentence, 7))
            #         continue
            #     p1, p2, p3, p4, p5, p6, times, p8, p9 = result.groups()
            #     part1 = '梯形{0}{1}{2}{3}中，'.format(p1, p4, p3, p2)
            #
            #     if times.is_Rational == True:  # times 肯定是正的;is_Rational 是有理数。
            #         t1, t2 = times.q, times.p
            #         t1 = '' if t1 == 1 else '{}'.format(t1)
            #         t2 = '' if t2 == 1 else '{}'.format(t2)
            #         if times.q > times.p:  # 不能出现2AB=CD，应该是CD=2AB
            #             part2 = '{2}{3}//{0}{1}且{5}{2}{3}={4}{0}{1}'.format(p5, p6, p8, p9, t1, t2)
            #             title1 = '{}{}'.format(part1, part2)
            #             title2 = part2
            #         else:
            #             part2 = '{0}{1}//{2}{3}且{4}{0}{1}={5}{2}{3}'.format(p5, p6, p8, p9, t1, t2)
            #             title1 = '{}{}'.format(part1, part2)
            #             title2 = part2
            #     else:  # 下面的代码应该没有执行
            #         part2 = '{0}{1}//{2}{3}且{0}{1}={4}{2}{3}'.format(p5, p6, p8, p9, times)
            #         title1 = '{}{}'.format(part1, part2)
            #         title2 = part2
            #     one_relation.append((title, 7))

            elif key == 'point_on_circle_from_diameter':  # point_on_circle_from_diameter(D,BC)
                pattern = compile(
                    r'point_on_circle_from_diameter\(([A-Z]),([A-Z])([A-Z])\)')
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 8))
                    continue

                p1, p2, p3 = result.groups()
                title = '点{}在以{}{}为直径的圆上'.format(p1, p2, p3)
                one_relation.append((title, 8))

            elif key == '4_points_on_circle':
                pattern = compile(
                    r'4_points_on_circle\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)')
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 8))
                    continue

                p1, p2, p3, p4 = result.groups()
                title = '{}、{}、{}、{}四点共圆'.format(p1, p2, p3, p4)
                one_relation.append((title, 8))

            elif key == 'equal':
                pattern = compile(
                    r'equal\((\S*),(\S*)\)')
                result = pattern.match(sentence)
                if (result == None):
                    one_relation.append((sentence, 10))
                    continue

                s1, s2 = result.groups()
                title = '{}={}'.format(s1, s2)
                one_relation.append((title, 10))

        if len(one_relation) > 1:
            word_list = []
            right = -100
            for word in one_relation:
                word_list.append(word[0])
                if right < word[1]:
                    right = word[1]
            if word_list[1] == '向外做':
                title = '由{}的两边{}{}、{}'.format(word_list[0], word_list[1], word_list[2], word_list[3])
            else:
                title = '，'.join(word_list)
            relation = (title, right)
            paragraph_sort.append(relation)

        elif len(one_relation) == 1:
            relation = one_relation[0]
            paragraph_sort.append(relation)

    # 形成题干
    gentenced_graph = []
    relation_pair = sorted(paragraph_sort, key=lambda x: x[1], reverse=False)
    for relations in relation_pair:
        gentenced_graph.append(relations[0])

    title = '，'.join(gentenced_graph)
    # title = '，{}. 求证：{}.'.format(title, paragraph_sort[-1][0])
    # title = '如图' + title

    title = re.sub(re.compile(r"([0-9]) \\cdot"), r'\1', title)  # 逗号再次精简
    return paragraph_sort, title


def Mapping(expression, new_points, hypothesis, conclusion, draw, session):
    '''

    :param expression: 点几何表达式
    :param new_points:  衍生点替换关系
    :param hypothesis:  题干语义，生成习题的题干文本
    :param conclusion:  结论
    :param draw:   画图语句，和new_points 共同生成坐标方程
    :param session: Mathematica会话
    :return:

    (0, message, img, hypothesis, basepoint, scale, isolated_point,picture_feature)
    作图成功标识，作图结果信息，矫正后的形式语言，基本点、图片比例、是否有孤立点
    scale -1图片点有重合，-2图片分布紧密但不行，1图片分布紧密但还可以，2图片分布比例完全合适。
    isolated_point：是否有孤立点，0表示没有, 1表示有
    picture_feature: 图片特征 点对的最小距离、最大距离、点的个数，用来区分相同的题目
    '''

    # print('0**************')
    # 首先调整衍生点的顺序
    derives = {}  # 衍生点字典，key为延伸点名称，值为列表（2元素，分别为：点几何形式的表达式，表达式包含的字母）
    draw_sentence = draw  # 画图语句来自 new_points和 draw
    for br in new_points.split(';'):
        if br == '':
            continue
        key, value = br.split('=')[0:2]
        letters = set(compile(r'[A-Z]').findall(value))
        derives[key] = [value, letters]
        draw_sentence = draw_sentence + value

    derive_point_list = list(derives.keys())
    derive_points = set(derives.keys())

    for ke in derives.keys():
        derives[ke][1] = derive_points.intersection(derives[ke][1])

    for ke, va in derives.items():
        # print(derive_point_list)
        letters = va[1]
        if len(letters) > 0:  # ke 应该在letters中所有元素的后面
            derive_point_list.remove(ke)
            max_inx = -1
            for le in letters:
                inx = derive_point_list.index(le)
                if max_inx < inx:
                    max_inx = inx
            derive_point_list.insert(max_inx + 1, ke)

    # 1.筛选出基本点和原点
    base_point = []  # 基本点
    cap_in_expression = ''.join(list(set(compile(r'[A-Z]').findall(expression))))  # 确保表达式中的字母在画图语句中存在
    all_capital = compile(r'[A-Z]').findall(draw_sentence + cap_in_expression)
    capital_times = sorted(Counter(all_capital).items(), key=lambda item: item[1], reverse=True)  # 按出现次数降序排列
    for point in capital_times:
        if point[0] not in derive_points:  # 不能要衍生点
            base_point.append(point[0])

    # 特殊情况原点
    alternative = []
    has_cyclic_quadrilateral = compile(
        r'cyclic_quadrilateral\(([A-Z]),quadrilateral\([A-Z],[A-Z],[A-Z],[A-Z]\)\)').search(draw)  # 四边形内接圆
    has_circumcenter = compile(r'circumcenter\(([A-Z]),triangle\([A-Z],[A-Z],[A-Z]\)\)').search(draw)  # 三角形垂心
    has_intersect = compile(r'line_intersection_at\([A-Z][A-Z],[A-Z][A-Z],([A-Z])\)').search(draw)  # 交点

    if has_cyclic_quadrilateral != None:
        po = has_cyclic_quadrilateral.groups()[0]
        if po not in alternative:
            alternative.append(po)

    if has_intersect != None:
        po = has_intersect.groups()[0]
        if po not in alternative:
            alternative.append(po)

    if has_circumcenter != None:
        po = has_circumcenter.groups()[0]
        if po not in alternative:
            alternative.append(po)

    if base_point[0] not in alternative:  # 出现次数第一多的字母
        alternative.append(base_point[0])
    if base_point[1] not in alternative:  # 出现次数第二多
        alternative.append(base_point[1])
    # print(basepoint)

    # 为了让图形保持端正，除原点外的某一个点的纵坐标也为0,横坐标不能等于0；这样处理也可以让坐标方程解的时候复杂度更低
    origin, co_point = alternative[0:2]
    # 指定原点
    # origin='A'
    # co_point='B'

    # 2.形成所有点的坐标和未知数列表
    point = {}  # 字典: 点名称-坐标
    unknown_list = []  # 未知数变量列表
    index = 1

    base_point = sorted(base_point)
    for p in base_point:
        if p == origin:
            point[p] = (sympify(0), sympify(0))
        elif co_point != None and p == co_point:
            coordinate_x = symbols('x{0}'.format(index))
            point[p] = (coordinate_x, sympify(0))
            unknown_list.append('x{0}'.format(index))
            index = index + 1
        else:
            coordinate_x = symbols('x{0}'.format(index))
            coordinate_y = symbols('y{0}'.format(index))
            point[p] = (coordinate_x, coordinate_y)
            unknown_list.append('x{0}'.format(index))
            unknown_list.append('y{0}'.format(index))
            index = index + 1

    # 形成衍生点的坐标
    for key in derive_point_list:
        value = sympify(derives[key][0], _clash)
        coordinate_x = value
        coordinate_y = value
        for s in value.free_symbols:
            coordinate_x = coordinate_x.subs(s, point[str(s)][0])
            coordinate_y = coordinate_y.subs(s, point[str(s)][1])
        point[key] = coordinate_x, coordinate_y

    # print(point)
    # print('2**************')

    # 3，构建坐标构造方程
    equation_list = []  # 等式和不等式列表列表
    # 额外增加条件
    # draw='{};{}'.format(draw,'acute(A,O,B)')
    # draw = '{};{}'.format(draw, 'acute(A,B,C)')
    # draw = '{};{}'.format(draw, 'obtuse(A,B,C)')
    # draw = '{};{}'.format(draw, 'vector_relation((C-B)^2-2*(B-O)^2)')
    # draw = '{};{}'.format(draw, 'inside_triangle(F,triangle(A,B,C))')
    # draw = '{};{}'.format(draw, 'both_side(A,C,BP)')
    # draw = '{};{}'.format(draw, 'inside_triangle(F,triangle(A,B,C))')
    # draw = '{};{}'.format(draw, 'vector_relation(0.64*(B-A)^2-(B-C)^2)')
    for sentence in draw.split(';'):
        if sentence.replace(' ', '') == '':
            continue
        sentence = sentence.strip()
        index = sentence.find('(')
        if (index == -1):
            exit(-1, '画图形式语言有错')

        key = sentence[0:index]

        if key == 'point':  # 为了零时处理点
            continue

        if key == 'quadrilateral':  # 四边形 parallelogram(A,B,C,D)
            result = compile(r'quadrilateral\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').search(sentence)
            A, B, C, D = result.groups()  # A B C D

            # (1)AB在DC同侧
            v1 = (point[A][0] - point[C][0]) * (point[A][1] - point[D][1]) - (point[A][0] - point[D][0]) * (
                    point[A][1] - point[C][1])  # 直线两点式
            v2 = (point[B][0] - point[C][0]) * (point[B][1] - point[D][1]) - (point[B][0] - point[D][0]) * (
                    point[B][1] - point[C][1])
            equation_new = '{}>0'.format(v1 * v2)
            equation_list.append(equation_new)

            # (2)AD在BC同侧
            v1 = (point[A][0] - point[C][0]) * (point[A][1] - point[B][1]) - (point[A][0] - point[B][0]) * (
                    point[A][1] - point[C][1])  # 直线两点式
            v2 = (point[D][0] - point[C][0]) * (point[D][1] - point[B][1]) - (point[D][0] - point[B][0]) * (
                    point[D][1] - point[C][1])
            equation_new = '{}>0'.format(v1 * v2)
            equation_list.append(equation_new)

            # (3)AC在BD两侧
            v1 = (point[A][0] - point[D][0]) * (point[A][1] - point[B][1]) - (point[A][0] - point[B][0]) * (
                    point[A][1] - point[D][1])  # 直线两点式
            v2 = (point[C][0] - point[D][0]) * (point[C][1] - point[B][1]) - (point[C][0] - point[B][0]) * (
                    point[C][1] - point[D][1])

            equation_new = '{}<0'.format(v1 * v2)
            equation_list.append(equation_new)


        elif key == 'notcollinear':  # 3点不共线  notcollinear(D,A,C)
            result = compile(r'([A-Z])').findall(sentence)
            p1, p2, p3 = result
            equation_list.append('({0}-({1}))*({2}-({3}))!=({4}-({5}))*({6}-({7}))'.format(point[p2][0], point[p1][0],
                                                                                           point[p3][1], point[p2][1],
                                                                                           point[p3][0], point[p2][0],
                                                                                           point[p2][1], point[p1][1]))
        elif key == 'collinear':  # 3点不共线  notcollinear(D,A,C)
            result = compile(r'([A-Z])').findall(sentence)
            p1, p2, p3 = result
            equation_list.append('({0}-({1}))*({2}-({3}))==({4}-({5}))*({6}-({7}))'.format(point[p2][0], point[p1][0],
                                                                                           point[p3][1], point[p2][1],
                                                                                           point[p3][0], point[p2][0],
                                                                                           point[p2][1], point[p1][1]))
        elif key == 'lieson':  # 点在线段上，先共线，再限定位置
            result = compile(r'lieson\(([A-Z]),([A-Z])([A-Z])\)').search(sentence)
            p1, p2, p3 = result.groups()
            equation_list.append('({0})*({1})==({2})*({3})'.format(point[p2][0] - point[p1][0],
                                                                   point[p3][1] - point[p2][1],
                                                                   point[p3][0] - point[p2][0],
                                                                   point[p2][1] - point[p1][1]))
            equation_list.append(
                '{0}>=Min[{2},{1}]'.format(point[p1][1], point[p2][1], point[p3][1]))
            equation_list.append(
                '{0}<=Max[{2},{1}]'.format(point[p1][1], point[p2][1], point[p3][1]))  # 这里不能要等号，要等号点就重合了，而且解方程时间长
            equation_list.append(
                '{0}>=Min[{2},{1}]'.format(point[p1][0], point[p2][0], point[p3][0]))
            equation_list.append(
                '{0}<=Max[{2},{1}]'.format(point[p1][0], point[p2][0], point[p3][0]))  # 这里不能要等号，要等号点就重合了，而且解方程时间长
            # equation_list.append(
            #     '{0}!={1}'.format(point[p2][0], point[p3][0]))

        elif key == 'line_intersection_at':  # 相交线  line_intersection_at(AE,CF,D)
            result = compile(r'line_intersection_at\(([A-Z])([A-Z]),([A-Z])([A-Z]),([A-Z])\)').search(sentence)
            p1, p2, p3, p4, p5 = result.groups()
            equation_new = '({0})*({1})==({2})*({3})'.format(point[p2][0] - point[p1][0],
                                                             point[p5][1] - point[p2][1],
                                                             point[p5][0] - point[p2][0],
                                                             point[p2][1] - point[p1][1])
            equation_list.append(equation_new)

            equation_new = '({0})*({1})==({2})*({3})'.format(point[p4][0] - point[p3][0],
                                                             point[p5][1] - point[p4][1],
                                                             point[p5][0] - point[p4][0],
                                                             point[p4][1] - point[p3][1])
            equation_list.append(equation_new)

        elif key == 'acute':  # acute(A,O,N) 锐角
            result = compile(r'acute\(([A-Z]),([A-Z]),([A-Z])\)').search(sentence)
            p1, p2, p3 = result.groups()
            equation_new = '({0})*({1})+({2})*({3})>0'.format(point[p1][0] - point[p2][0], point[p3][0] - point[p2][0],
                                                              point[p1][1] - point[p2][1], point[p3][1] - point[p2][1])
            equation_list.append(equation_new)

        elif key == 'angle_equal':  # angle_equal(AON) 角度；特殊角度值  60度 30度
            result = compile(r'angle_equal\(([A-Z]),([A-Z]),([A-Z]),([0-9/-]*)\)').search(sentence)
            p1, p2, p3, value = result.groups()
            # 角的两边作为两个向量
            V1 = (point[p1][0] - point[p2][0], point[p1][1] - point[p2][1])
            V2 = (point[p3][0] - point[p2][0], point[p3][1] - point[p2][1])
            equation_new = '{0}+{1}==Sqrt[{2}]*Sqrt[{3}]*({4})'.format(V1[0] * V2[0], V1[1] * V2[1],
                                                                       (V1[0]) ** 2 + (V1[1]) ** 2,
                                                                       (V2[0]) ** 2 + (V2[1]) ** 2, value)
            equation_list.append(equation_new)

        elif key == 'angle_same':  # angle_same(A,B,P,A,C,P)  角度相等
            result = compile(r'angle_same\(([A-Z]),([A-Z]),([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').search(sentence)
            p1, p2, p3, p4, p5, p6 = result.groups()
            # 角的两边作为两个向量
            V1 = (point[p1][0] - point[p2][0], point[p1][1] - point[p2][1])
            V2 = (point[p3][0] - point[p2][0], point[p3][1] - point[p2][1])

            V3 = (point[p4][0] - point[p5][0], point[p4][1] - point[p5][1])
            V4 = (point[p6][0] - point[p5][0], point[p6][1] - point[p5][1])
            equation_new = '{0}*Sqrt[{1}]*Sqrt[{2}]=={3}*Sqrt[{4}]*Sqrt[{5}]'.format(V1[0] * V2[0] + V1[1] * V2[1],
                                                                                     (V1[0]) ** 2 + (V1[1]) ** 2,
                                                                                     (V2[0]) ** 2 + (V2[1]) ** 2,
                                                                                     V3[0] * V4[0] + V3[1] * V4[1],
                                                                                     (V3[0]) ** 2 + (V3[1]) ** 2,
                                                                                     (V4[0]) ** 2 + (V4[1]) ** 2)
            equation_list.append(equation_new)

        elif key == 'parallel':  # 线段平行，向量平行
            result = compile(r'parallel\(([A-Z])([A-Z]),([A-Z])([A-Z])\)').search(sentence)
            p1, p2, p3, p4 = result.groups()  # 前三个点不共线
            equation_list.append('({0})*({1})!=({2})*({3})'.format(point[p2][0] - point[p1][0],
                                                                   point[p3][1] - point[p2][1],
                                                                   point[p3][0] - point[p2][0],
                                                                   point[p2][1] - point[p1][1]))

            equation_list.append(
                '({0})*({1})==({2})*({3})'.format(point[p2][0] - point[p1][0], point[p4][1] - point[p3][1],
                                                  point[p4][0] - point[p3][0], point[p2][1] - point[p1][1]))

        elif key == 'not_parallel':  # 线段不平行
            result = compile(r'not_parallel\(([A-Z])([A-Z]),([A-Z])([A-Z])\)').search(sentence)
            p1, p2, p3, p4 = result.groups()
            equation_list.append('({0})*({1})!=({2})*({3})'.format(point[p2][0] - point[p1][0],
                                                                   point[p4][1] - point[p3][1],
                                                                   point[p4][0] - point[p3][0],
                                                                   point[p2][1] - point[p1][1]))

        elif key == 'line_less':  # 线段长度关系
            result = compile(r'line_less\(([A-Z])([A-Z]),([A-Z])([A-Z])\)').search(sentence)
            p1, p2, p3, p4 = result.groups()
            equation_list.append(
                '{0}<{1}'.format((point[p1][0] - point[p2][0]) ** 2 + (point[p1][1] - point[p2][1]) ** 2,
                                 (point[p3][0] - point[p4][0]) ** 2 + (point[p3][1] - point[p4][1]) ** 2))

        elif key == 'equal':  # 线段相等  equal(HD,DE)
            pattern = compile(
                r'equal\((\S*),(\S*)\)')
            part1, part2 = pattern.match(sentence).groups()
            if compile(r'[\^\-\+]').search(sentence) == None:  # 只是两个线段之间的关系,需要转换为平方关系
                part1, part2 = sympify(part1, _clash), sympify(part2, _clash)
                vector = str(expand(part1 ** 2 - part2 ** 2))

            else:
                vector = str(expand(sympify('{}-({})'.format(part1, part2), _clash)))

            for square in set(compile(r'([A-Z][A-Z]\*\*2)').findall(vector)):  # 乘积
                p1, p2 = compile(r'([A-Z])([A-Z])').search(square).groups()  # 那个大写字母
                text = '({})'.format((point[p1][0] - point[p2][0]) ** 2 + (point[p1][1] - point[p2][1]) ** 2)
                vector = vector.replace(square, text)

            for mul in set(compile(r'([A-Z][A-Z]\*[A-Z][A-Z])').findall(vector)):  # 乘积
                p1, p2, p3, p4 = compile(r'([A-Z])([A-Z])\*([A-Z])([A-Z])').search(
                    mul).groups()  # 这里有问题，默认线段共线或者平行
                text = '(Abs[{}])'.format(point[p1][0] * point[p2][0] + point[p1][1] * point[p2][1])
                vector = vector.replace(mul, text)

            equation_list.append('{}==0'.format(vector))

        elif key == 'notequal':  # 线段不相等  notequalequal(HD,DE)
            result = compile(r'equal\(([A-Z])([A-Z]),([A-Z])([A-Z])\)').search(sentence)
            p1, p2, p3, p4 = result.groups()
            equation_list.append('{0}+{1}!={2}+{3}'.format((point[p1][0] - point[p2][0]) ** 2,
                                                           (point[p1][1] - point[p2][1]) ** 2,
                                                           (point[p3][0] - point[p4][0]) ** 2,
                                                           (point[p3][1] - point[p4][1]) ** 2))

        elif key == '4_points_on_circle':  # 4点共圆  4_points_on_circle(A,E,C,F)
            result = compile(r'4_points_on_circle\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').search(sentence)
            p1, p2, p3, p4 = result.groups()
            p11, p12, p21, p22, p31, p32, p41, p42 = sympify(
                [point[p1][0], point[p1][1], point[p2][0], point[p2][1], point[p3][0], point[p3][1], point[p4][0],
                 point[p4][1]])
            M = Matrix([[p11 ** 2 + p12 ** 2, p11, p12, 1], [p21 ** 2 + p22 ** 2, p21, p22, 1],
                        [p31 ** 2 + p32 ** 2, p31, p32, 1], [p41 ** 2 + p42 ** 2, p41, p42, 1]])
            equation_list.append('{}==0'.format(M.det()))

        elif key == 'perpendicular':  # 垂直  perpendicular(AD,BC)
            result = compile(r'perpendicular\(([A-Z])([A-Z]),([A-Z])([A-Z])\)').search(sentence)
            p1, p2, p3, p4 = result.groups()
            equation_list.append(
                '{0}+{1}==0'.format((point[p1][0] - point[p2][0]) * (point[p3][0] - point[p4][0]),
                                    (point[p1][1] - point[p2][1]) * (point[p3][1] - point[p4][1])))
        elif key == 'notperpendicular':  # 垂直  perpendicular(AD,BC)
            result = compile(r'perpendicular\(([A-Z])([A-Z]),([A-Z])([A-Z])\)').search(sentence)
            p1, p2, p3, p4 = result.groups()
            equation_list.append(
                '{0}+{1}!=0'.format((point[p1][0] - point[p2][0]) * (point[p3][0] - point[p4][0]),
                                    (point[p1][1] - point[p2][1]) * (point[p3][1] - point[p4][1])))

        elif key == 'orthocenter':  # 三角形垂心
            result = compile(
                r'orthocenter\(([A-Z]),triangle\(([A-Z]),([A-Z]),([A-Z])\)\)').search(sentence)
            p1, p2, p3, p4 = result.groups()
            # 三点不共线
            equation_list.append('({0})*({1})!=({2})*({3})'.format(point[p3][0] - point[p2][0],
                                                                   point[p4][1] - point[p3][1],
                                                                   point[p4][0] - point[p3][0],
                                                                   point[p3][1] - point[p2][1]))
            equation_list.append(
                '{0}+({1})==0'.format((point[p1][0] - point[p2][0]) * (point[p3][0] - point[p4][0]),
                                      (point[p1][1] - point[p2][1]) * (point[p3][1] - point[p4][1])))

            equation_list.append(
                '{0}+({1})==0'.format((point[p1][0] - point[p3][0]) * (point[p2][0] - point[p4][0]),
                                      (point[p1][1] - point[p3][1]) * (point[p2][1] - point[p4][1])))

        elif key == 'circumcenter':  # 三角形外心 orthocenter(O,triangle(A,B,C))
            result = compile('circumcenter\(([A-Z]),triangle\(([A-Z]),([A-Z]),([A-Z])\)\)').match(sentence)
            p1, p2, p3, p4 = result.groups()
            equation_list.append(
                '{}=={}'.format((point[p1][0] - point[p2][0]) ** 2 + (point[p1][1] - point[p2][1]) ** 2,
                                (point[p1][0] - point[p3][0]) ** 2 + (point[p1][1] - point[p3][1]) ** 2))
            equation_list.append(
                '{}=={}'.format((point[p1][0] - point[p3][0]) ** 2 + (point[p1][1] - point[p3][1]) ** 2,
                                (point[p1][0] - point[p4][0]) ** 2 + (point[p1][1] - point[p4][1]) ** 2))

        elif key == 'cyclic_quadrilateral':  # 圆内接四边形
            result = compile('cyclic_quadrilateral\(([A-Z]),quadrilateral\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)\)').search(
                sentence)
            p1, p2, p3, p4, p5 = result.groups()
            equation_list.append(
                '{}+{}=={}+{}'.format((point[p1][0] - point[p2][0]) ** 2,
                                      (point[p1][1] - point[p2][1]) ** 2,
                                      (point[p1][0] - point[p3][0]) ** 2,
                                      (point[p1][1] - point[p3][1]) ** 2))

            equation_list.append(
                '{}+{}=={}+{}'.format((point[p1][0] - point[p2][0]) ** 2,
                                      (point[p1][1] - point[p2][1]) ** 2,
                                      (point[p1][0] - point[p4][0]) ** 2,
                                      (point[p1][1] - point[p4][1]) ** 2))
            equation_list.append(
                '{}+{}=={}+{}'.format((point[p1][0] - point[p2][0]) ** 2,
                                      (point[p1][1] - point[p2][1]) ** 2,
                                      (point[p1][0] - point[p5][0]) ** 2,
                                      (point[p1][1] - point[p5][1]) ** 2))

        elif key == 'inside':  # 点在四边形内部
            result = compile('inside\(([A-Z]),quadrilateral\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)\)').search(
                sentence)
            p5, p1, p2, p3, p4 = result.groups()
            equation_list.append('Min[{0},{1},{2},{3}]<{4}'.format(point[p1][1], point[p2][1],
                                                                   point[p3][1], point[p4][1], point[p5][1]))

            equation_list.append('Max[{0},{1},{2},{3}]>{4}'.format(point[p1][1], point[p2][1],
                                                                   point[p3][1], point[p4][1], point[p5][1]))

            equation_list.append('Min[{0},{1},{2},{3}]<{4}'.format(point[p1][0], point[p2][0],
                                                                   point[p3][0], point[p4][0], point[p5][0]))

            equation_list.append('Max[{0},{1},{2},{3}]>{4}'.format(point[p1][0], point[p2][0],
                                                                   point[p3][0], point[p4][0], point[p5][0]))

        elif key == 'inside_triangle':  # 点在三角形内部  inside_triangle(F,triangle(A,H,G))
            result = compile('inside_triangle\(([A-Z]),triangle\(([A-Z]),([A-Z]),([A-Z])\)\)').search(
                sentence)
            p1, p2, p3, p4 = result.groups()

            # p1,p2在p3p4同侧
            v1 = (point[p1][0] - point[p3][0]) * (point[p1][1] - point[p4][1]) - (point[p1][0] - point[p4][0]) * (
                    point[p1][1] - point[p3][1])  # 直线两点式
            v2 = (point[p2][0] - point[p3][0]) * (point[p2][1] - point[p4][1]) - (point[p2][0] - point[p4][0]) * (
                    point[p2][1] - point[p3][1])
            equation_new = '{}>0'.format(v1 * v2)
            equation_list.append(equation_new)

            # p1,p3在p2p4同侧
            v1 = (point[p1][0] - point[p2][0]) * (point[p1][1] - point[p4][1]) - (point[p1][0] - point[p4][0]) * (
                    point[p1][1] - point[p2][1])  # 直线两点式
            v2 = (point[p3][0] - point[p2][0]) * (point[p3][1] - point[p4][1]) - (point[p3][0] - point[p4][0]) * (
                    point[p3][1] - point[p2][1])
            equation_new = '{}>0'.format(v1 * v2)
            equation_list.append(equation_new)

            # p1,p4在p3p2同侧
            v1 = (point[p1][0] - point[p3][0]) * (point[p1][1] - point[p2][1]) - (point[p1][0] - point[p2][0]) * (
                    point[p1][1] - point[p3][1])
            v2 = (point[p4][0] - point[p3][0]) * (point[p4][1] - point[p2][1]) - (point[p4][0] - point[p2][0]) * (
                    point[p4][1] - point[p3][1])
            equation_new = '{}>0'.format(v1 * v2)
            equation_list.append(equation_new)

        elif (key == 'point_on_circle_from_diameter'):  # 点在直径圆上 point_on_circle_from_diameter(AB)
            result = compile('point_on_circle_from_diameter\(([A-Z]),([A-Z])([A-Z])\)').match(
                sentence)
            p1, p2, p3 = result.groups()
            centre = ((point[p2][0] + point[p3][0]) / 2, (point[p2][1] + point[p3][1]) / 2)  # 圆心坐标
            equation_list.append(
                '{}=={}'.format((point[p1][0] - centre[0]) ** 2 + (point[p1][1] - centre[1]) ** 2,
                                (point[p2][0] - centre[0]) ** 2 + (point[p2][1] - centre[1]) ** 2))
        elif (key == 'square'):  # 正方形 square(O,B,C,E)
            result = compile('square\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').match(
                sentence)
            p1, p2, p3, p4 = result.groups()  # O,B,C,E

            # 平行四边形 E=O-B+C
            equation_list.append(
                '{0}=={1}'.format(point[p4][0], point[p1][0] - point[p2][0] + point[p3][0]))
            equation_list.append(
                '{0}=={1}'.format(point[p4][1], point[p1][1] - point[p2][1] + point[p3][1]))

            # 一对邻边垂直且相等  OB OE  12  14
            equation_list.append(
                '{0}=={1}'.format((point[p1][0] - point[p2][0]) ** 2 + (point[p1][1] - point[p2][1]) ** 2,
                                  (point[p1][0] - point[p4][0]) ** 2 + (point[p1][1] - point[p4][1]) ** 2))
            equation_list.append(
                '{0}+({1})==0'.format((point[p1][0] - point[p2][0]) * (point[p1][0] - point[p4][0]),
                                      (point[p1][1] - point[p2][1]) * (point[p1][1] - point[p4][1])))


        elif (key == 'regular_triangle'):  # 正三角形
            result = compile('regular_triangle\(([A-Z]),([A-Z]),([A-Z])\)').match(
                sentence)
            p1, p2, p3 = result.groups()
            # 3点不共线
            equation_list.append(
                '({0})*({1})!=({2})*({3})'.format(point[p2][0] - point[p1][0], point[p3][1] - point[p2][1],
                                                  point[p3][0] - point[p2][0], point[p2][1] - point[p1][1]))
            # 2个相等
            equation_list.append(
                '{}=={}'.format((point[p1][0] - point[p2][0]) ** 2 + (point[p1][1] - point[p2][1]) ** 2,
                                (point[p1][0] - point[p3][0]) ** 2 + (point[p1][1] - point[p3][1]) ** 2))
            equation_list.append(
                '{}=={}'.format((point[p1][0] - point[p2][0]) ** 2 + (point[p1][1] - point[p2][1]) ** 2,
                                (point[p2][0] - point[p3][0]) ** 2 + (point[p2][1] - point[p3][1]) ** 2))

        elif (key == 'obtuse'):  # 钝角 obtuse(E,O,A)
            result = compile('obtuse\(([A-Z]),([A-Z]),([A-Z])\)').match(sentence)
            p1, p2, p3 = result.groups()
            equation_new = '({0})+({1})<0'.format((point[p1][0] - point[p2][0]) * (point[p3][0] - point[p2][0]),
                                                  (point[p1][1] - point[p2][1]) * (point[p3][1] - point[p2][1]))
            equation_list.append(equation_new)
        elif (key == 'both_side'):  # same_side(A,B,CD) A、B在直线CD的两侧
            result = compile('both_side\(([A-Z]),([A-Z]),([A-Z])([A-Z])\)').match(sentence)
            p1, p2, p3, p4 = result.groups()
            v1 = (point[p1][0] - point[p3][0]) * (point[p1][1] - point[p4][1]) - (point[p1][0] - point[p4][0]) * (
                    point[p1][1] - point[p3][1])  # 直线两点式
            v2 = (point[p2][0] - point[p3][0]) * (point[p2][1] - point[p4][1]) - (point[p2][0] - point[p4][0]) * (
                    point[p2][1] - point[p3][1])
            equation_new = '{}<0'.format(v1 * v2)
            equation_list.append(equation_new)
        elif (key == 'same_side'):  # same_side(A,B,CD) A、B在直线CD的同侧
            result = compile('same_side\(([A-Z]),([A-Z]),([A-Z])([A-Z])\)').match(sentence)
            p1, p2, p3, p4 = result.groups()
            v1 = (point[p1][0] - point[p3][0]) * (point[p1][1] - point[p4][1]) - (point[p1][0] - point[p4][0]) * (
                    point[p1][1] - point[p3][1])  # 直线两点式
            v2 = (point[p2][0] - point[p3][0]) * (point[p2][1] - point[p4][1]) - (point[p2][0] - point[p4][0]) * (
                    point[p2][1] - point[p3][1])
            equation_new = '{}>0'.format(v1 * v2)
            equation_list.append(equation_new)

        else:
            print('有未识别的坐标方程转换语句', sentence)

    # 增加额外的条件，画特殊的图形
    # 限定Y轴方向
    # p1, p2, p3 = 'B', 'A', 'P'
    # equation_list.append('Min[{0},{1}]<{2}'.format(point[p1][1], point[p2][1], point[p3][1]))
    # equation_list.append('Max[{0},{1}]>{2}'.format(point[p1][1], point[p2][1], point[p3][1]))

    # # 线段长度关系
    # p1, p2, p3, p4 = 'C', 'B', 'B', 'A'
    # equation_list.append('{0}+{1}!={2}+{3}'.format((point[p1][0] - point[p2][0]) ** 2,
    #                                                (point[p1][1] - point[p2][1]) ** 2,
    #                                                (point[p3][0] - point[p4][0]) ** 2,
    #                                                (point[p3][1] - point[p4][1]) ** 2))

    # # 线段长度关系 比例关系
    # p1, p2, p3, p4 = 'C', 'A', 'A', 'B'
    # equation_list.append('({0}+{1})/({2}+{3})==0.8**2'.format((point[p1][0] - point[p2][0]) ** 2,
    #                                                (point[p1][1] - point[p2][1]) ** 2,
    #                                                (point[p3][0] - point[p4][0]) ** 2,
    #                                                (point[p3][1] - point[p4][1]) ** 2))

    # 不共线
    # p1, p2, p3 = 'A', 'E', 'B'
    # equation_list.append('({0}-({1}))*({2}-({3}))!=({4}-({5}))*({6}-({7}))'.format(point[p2][0], point[p1][0],
    #                                                                                point[p3][1], point[p2][1],
    #                                                                                point[p3][0], point[p2][0],
    #                                                                                point[p2][1], point[p1][1]))
    # p1, p2, p3 = 'B', 'C', 'F'
    # equation_list.append('({0}-({1}))*({2}-({3}))==({4}-({5}))*({6}-({7}))'.format(point[p2][0], point[p1][0],
    #                                                                                point[p3][1], point[p2][1],
    #                                                                                point[p3][0], point[p2][0],
    #                                                                                point[p2][1], point[p1][1]))
    # 点在左边
    # p1,p2='A','P'
    # equation_list.append('{0}<{1}'.format(point[p1][0], point[p2][0]))
    # # #
    # # # #点在上面
    # p1, p2 = 'A', 'C'
    # equation_list.append('{0}<{1}'.format(point[p1][1], point[p2][1]))
    # # 不垂直
    # p1, p2, p3, p4 = 'A','B','B','C'
    # equation_list.append(
    #     '{0}+{1}!=0'.format((point[p1][0] - point[p2][0]) * (point[p3][0] - point[p4][0]),
    #                         (point[p1][1] - point[p2][1]) * (point[p3][1] - point[p4][1])))

    # p1, p2, p3, p4 = 'A','B','C','P' # P在ABD 三点确定的圆内部
    # x0, y0 = get_circle_symbolic(point[p1][0], point[p1][1], point[p2][0], point[p2][1], point[p3][0],
    #                              point[p3][1])
    # equation_list.append(
    #     '{0}>{1}'.format((point[p1][0] - x0) ** 2 + (point[p1][1] - y0) ** 2,
    #                      (point[p4][0] - x0) ** 2 + (point[p4][1] -y0) ** 2))

    # 组建mathematica 语句
    system_str = ','.join(equation_list)
    unknown_str = ','.join(unknown_list)
    wolframc_str = 'Clear["Global`*"];R1=FindInstance[{{{0}}}, {{{1}}},Reals]; R2={{{1}}} /.R1;N[R2,5]'.format(
        system_str, unknown_str)
    solve_result = solve_equations(wolframc_str, session)

    if (solve_result[0] < 0):  # 出错或者超时
        return (-1, '第一次解方程_' + solve_result[1])
    elif (solve_result[0] == 1):  # 无解，再不管退化的问题了
        return (-10, '第一次解方程无解，函数返回')

    # 5.有解，但是点的距离是否满足要求，还需要判断
    if (solve_result[0] == 0):
        # 5.1 有解,根据解得到点的坐标值
        unknown_subs = {}  # 未知数替换表
        data = solve_result[2][0]
        # if len(data)!=len(unknown_list):
        #     return (-1, '函数有解,但解的个数和未知数的个数不匹配')
        for i in range(0, len(unknown_list)):
            unknown_subs[unknown_list[i]] = data[i]
        point_draw = {}  # 所有点的坐标值,字典
        for p in point.keys():
            coordinate_x = point[p][0].subs(unknown_subs)
            coordinate_y = point[p][1].subs(unknown_subs)
            point_draw[p] = (coordinate_x, coordinate_y)

        # 5.2 判断解是否满足条件，保证图形中各部分内容分布比例适中
        success = None  # 0 满足条件的解  -1 不满足条件的解  -2无解
        dis_list = {}  # 所有点对之间的距离
        min_dis = float("inf")  # 最小距离
        max_dis = float("-inf")  # 最大距离
        for pair in combinations(list(point.keys()), 2):
            p1, p2 = point_draw[pair[0]], point_draw[pair[1]]
            dis = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
            dis_list[(pair[0], pair[1])] = dis
            if dis < min_dis:
                min_dis = dis
                min_pair = (pair[0], pair[1])
            if dis > max_dis:
                max_dis = dis
                max_pair = (pair[0], pair[1])
        #  5.2.1 点的距离满足条件，返回
        history_good = None

        if max_dis != 0 and min_dis / max_dis >= 1 / 100:
            success = (0, '第一次解方程得到满足条件的解', (min_dis, max_dis, min_pair, max_pair, [], point_draw, []))
        else:  # 5.2.2 点的距离不满足条件, 开始循环尝试
            if max_dis > 0 and min_dis > 0:  # 将满足条件的点对从dis_list删除
                for key in list(dis_list.keys()):
                    value = dis_list[key]
                    if value * 100 >= max_dis:
                        del dis_list[key]
            elif min_dis == 0:
                for key in list(dis_list.keys()):
                    value = dis_list[key]
                    if value > 0:  # 当前这对节点满足要求,从这里删除
                        del dis_list[key]

            best = (min_dis, max_dis, min_pair, max_pair, equation_list.copy(), point_draw,
                    dis_list)  # best记录随机过程中效果最好的那次。equation_list存储的是方程组，其他的变量记录的是在 equation_list 方程组下相应的结果; 后面每次随机中用的方程都是在equation_list 的基础上添加的。
            best_old = copy.deepcopy(best)  # best_old存储第一次解方程后的结果
            wolframc_str_fail = {}  # 将每次的结果存下来，为了跳过循环，节省时间
            for i in range(0, 20):
                # 随机20次； 每次进来后以 best 中的信息为基础进行操作，所以循环前和每次循环中要把best 的值修改好
                # print('点的距离不能满足要求，开始多次随机尝试','第{}次'.format(str(i)))
                # print(success)

                min_dis, max_dis, min_pair, max_pair, equation_list, point_draw, dis_list = best
                equation_append = []
                # 5.2.2.1  添加进一步优化的条件,并进一步解方程
                if min_dis <= 1e-10:  # 5.2.1.1  有重合点，先让点不重合
                    key_list = random.sample(dis_list.keys(), min(len(dis_list.keys()), 2))  # 不要一次性约束所有的重叠点，容易造成超时和无解
                    key_list = sorted(key_list, key=lambda x: x)
                    for key in key_list:
                        value = dis_list[key]
                        if value == 0:
                            p1, p2 = point[key[0]], point[key[1]]
                            ran = random.randint(0, 1)  # 随机放这里是不是最好
                            if (ran == 1):  # 或者让x坐标不相等，或者让y坐标不相等 ,随机添加
                                equation_append.append('{}!={}'.format(p1[0], p2[0]))  # 不需要随机 X 和 Y 因为如果游街就肯定能满足
                            else:
                                equation_append.append('{}!={}'.format(p1[1], p2[1]))
                else:  # 没有重合点, 随机选择一个不满足条件的点进行优化
                    if history_good == None:
                        history_good = copy.deepcopy(best)
                    elif history_good[0] / history_good[1] < best[0] / best[1]:
                        history_good = copy.deepcopy(best)
                    if len(dis_list.keys()) > 0:
                        key_list = list(dis_list.keys())
                        key = random.sample(key_list, 1)[0]
                        p1, p2 = point[key[0]], point[key[1]]
                        p3, p4 = point[max_pair[0]], point[max_pair[1]]
                        max_dis_str = '({}-({}))^2+({}-({}))^2'.format(str(p3[0]), str(p4[0]), str(p3[1]), str(p4[1]))
                        equation_append.append('(({}-({}))^2+({}-({}))^2)*100>={}'.format(p1[0], p2[0],
                                                                                          p1[1], p2[1], max_dis_str))
                    else:
                        success = (-2, '有解，但不满足要求_1', best)

                # print(best)
                new_equation_list = equation_list + equation_append  # 这3个列表没有共享内存
                equation_append_str = str(
                    new_equation_list[len(best_old[4]):len(new_equation_list)])  # 后面每次随机中用的方程都是在原始方程的的基础上添加的。
                if equation_append_str in wolframc_str_fail.keys():
                    # print('命中')
                    best = wolframc_str_fail[equation_append_str]
                    continue

                system_str = ','.join(equation_list + equation_append)
                wolframc_str = 'Clear["Global`*"];R1=FindInstance[{{{0}}}, {{{1}}},Reals]; R2={{{1}}} /.R1;N[R2, 5]'.format(
                    system_str, unknown_str)
                solve_result = solve_equations(wolframc_str, session)

                # 5.2.2.2 再一次解方程后判断结果
                if solve_result[0] == -2:  # 超时返回到最初的状态，然后在解空间中继续随机寻找！  一定要这样写！！！！！！！！
                    best = copy.deepcopy(best_old)
                    # best[4].pop()
                    success = (-2, '循环中解方程超时', best)
                    if equation_append_str not in wolframc_str_fail.keys():
                        wolframc_str_fail[equation_append_str] = copy.deepcopy(best_old)
                    continue
                elif solve_result[0] < 0:  # 出错返回
                    return (-2, '点的距离不满足要求后，多次循环尝试时，解约束系统出错_' + solve_result[1])
                elif solve_result[0] == 1:  # 无解,best回到最初的状态,重新通过随机在解空间中寻找值
                    best = copy.deepcopy(best_old)  # 无解就回到最初的状态，这样才能通过随机在解空间中寻找！  一定要这样写！！！！！！！！
                    success = (-2, '多次尝试解方程时无解', best)
                    if equation_append_str not in wolframc_str_fail.keys():
                        wolframc_str_fail[equation_append_str] = copy.deepcopy(
                            best_old)  # 不能直接是best_old,不然best_old 会被修改
                    continue
                elif solve_result[0] == 0:  # 有解，还要判断
                    # 1).根据解得到点的坐标值
                    unknown_subs = {}  # 未知数替换表
                    data = solve_result[2][0]
                    for ii in range(0, len(unknown_list)):
                        unknown_subs[unknown_list[ii]] = data[ii]
                    point_draw = {}  # 所有点的坐标值,字典
                    for p in point.keys():
                        coordinate_x = point[p][0].subs(unknown_subs)
                        coordinate_y = point[p][1].subs(unknown_subs)
                        point_draw[p] = (coordinate_x, coordinate_y)

                    # 2).判断是否满足要求
                    dis_list = {}  # 所有点对之间的距离
                    min_dis = float("inf")  # 最小距离
                    max_dis = float("-inf")  # 最大距离
                    for pair in combinations(list(point.keys()), 2):
                        p1, p2 = point_draw[pair[0]], point_draw[pair[1]]
                        dis = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                        dis_list[(pair[0], pair[1])] = dis
                        if dis < min_dis:
                            min_dis = dis
                            min_pair = (pair[0], pair[1])
                        if dis > max_dis:
                            max_dis = dis
                            max_pair = (pair[0], pair[1])

                    if max_dis != 0 and min_dis / max_dis >= 1 / 100:  # 3) 点的距离满足要求，返回
                        best = (
                            min_dis, max_dis, min_pair, max_pair, equation_list.copy(), point_draw, dis_list)  # 更新best
                        success = (0, '{}次尝试后得到满足条件的解'.format(str(i + 1)), best)
                        break

                    elif max_dis == 0 or (max_dis != 0 and min_dis / max_dis < 1 / 100):  # 4) 不满足条件,分情况判断
                        ratio = -1 if max_dis == 0 else min_dis / max_dis  # 所有点重合在一起,比值是负的
                        best_ratio = -1 if best[1] == 0 else best[0] / best[1]  # 历史最好

                        if ratio >= best_ratio:  # 4.1)  当前结果比best好，更新best ; 这里条件大于等于，尤其是等于，当前状态和best效果一样，要把best更新到当前状态，因为当前状态的约束体条件比best的约束条件多
                            if max_dis > 0 and min_dis > 0:  # 当前这对节点满足要求,从这里删除
                                for key in list(dis_list.keys()):
                                    value = dis_list[key]
                                    if value * 100 >= max_dis:
                                        del dis_list[key]
                            elif min_dis == 0:
                                for key in list(dis_list.keys()):
                                    value = dis_list[key]
                                    if value > 0:
                                        del dis_list[key]
                            new_equation_list = equation_list + equation_append
                            best = (min_dis, max_dis, min_pair, max_pair, new_equation_list, point_draw,
                                    dis_list)
                            success = (-1, '多次尝试后，有解，但不满足条件_2', best)
                            if equation_append_str not in wolframc_str_fail.keys():
                                wolframc_str_fail[equation_append_str] = copy.deepcopy(
                                    best)
                        else:  # 4.2)  当前结果比best差， 恢复到best，开始搞原始最好的的 次小 ；有3个值 当前结果，当前best,原始best(best_old)
                            success = (-1, '多次尝试后，有解，但不满足条件_3', best)
                            if equation_append_str not in wolframc_str_fail.keys():
                                wolframc_str_fail[equation_append_str] = copy.deepcopy(
                                    best)

    if success[0] < 0:  # 20230320修改
        return (success[0], success[1])

    message, min_dis, max_dis, point_draw, dis_list = success[1], success[2][0], success[2][1], success[2][5], \
                                                      success[2][6]
    point_draw_old = point_draw.copy()  # 点的原始坐标，在计算点是否共线的时候用，因为point_draw 后面四舍五入，精度会有问题

    if history_good != None:  # best与历史的坐标解（history_good）比较，如果history_good好，则用history_good
        h_min_dis, h_max_dis, h_point_draw, h_dist_dis_list = history_good[0], history_good[1], history_good[5], \
                                                              history_good[6]
        if best[0] == 0:
            min_dis, max_dis, point_draw, dis_list = h_min_dis, h_max_dis, h_point_draw, h_dist_dis_list
        elif history_good[0] / history_good[1] > best[0] / best[1]:
            min_dis, max_dis, point_draw, dis_list = h_min_dis, h_max_dis, h_point_draw, h_dist_dis_list

    # 验证结果是否正确,看各表达式是否为0
    error = 0
    for expre in expression.split(';'):
        if expre == '':
            continue
        expre = str(expand(sympify(expre, _clash)))

        for square in set(compile(r'([A-Z]\*\*2)').findall(expre)):  # 乘积
            result = compile(r'([A-Z])').findall(square)  # 那个大写字母
            p = result[0]
            text = '(' + str(point_draw[p][0] ** 2 + point_draw[p][1] ** 2) + ')'
            expre = expre.replace(square, text)
        for mul in set(compile(r'([A-Z]\*[A-Z])').findall(expre)):  # 乘积
            result = compile(r'([A-Z])').findall(mul)  # 那个大写字母
            p1, p2 = result[0], result[1]
            text = '(' + str(point_draw[p1][0] * point_draw[p2][0] + point_draw[p1][1] * point_draw[p2][1]) + ')'
            expre = expre.replace(mul, text)
        # print(round(float(expand(sympify(expre, _clash) * 1000))))  # 只要大概准确就行，这种方法盘飧是否正确，张院士他们验证了
        error += round(float(expand(sympify(expre, _clash) * 1000)))
    if error > 1:
        return (-1, '误差检验未通过')

    # 6.开始作图
    ##6.1 确定图形的缩放和平移
    # y加1 向下移动；x加1 向右移动;图形最上面距离边框50像素，图形最左边具体边框50像素；

    img = np.zeros((400, 400, 3), dtype='uint8')  # 图形数据和numpy数组对应关系要求 Y和X 要替换，
    img[:, :] = (255, 255, 255)  # 图形底色为全白

    # 先计算图形的凸包中心
    my_conhull = convex_hull(list(point_draw.values()))  # 计算图形的中心

    center_point = [0, 0]
    for i in range(len(my_conhull)):
        center_point[0] += my_conhull[i][0]
        center_point[1] += my_conhull[i][1]

    center_point[0] /= len(my_conhull)
    center_point[1] /= len(my_conhull)

    max_radius = float("-inf")
    for i in range(len(my_conhull)):
        radius = (center_point[0] - my_conhull[i][0]) ** 2 + (center_point[1] - my_conhull[i][1]) ** 2
        if max_radius < radius:
            max_radius = radius

    bl = 160 / math.sqrt(max_radius)  # 缩放比例
    va_x = 200 - center_point[0] * bl  # 向左平移量
    va_y = 200 - center_point[1] * bl  # 向下平移量

    for key, value in point_draw.items():  # 坐标修正
        point_draw[key] = (int(round(bl * value[0] + va_x)), int(round(bl * value[1] + va_y)))  # 将最左边的点移到（50，50）

    # 画图的一些配置选项
    black = (0, 0, 0)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    thickness = 2  # 线、圆轮廓厚度，可以为 0 、4、8
    lineType = cv.LINE_AA  # 线段类型

    # 6.2 图片元素分布比例检查
    # print(min_dis, max_dis, len(point))
    scale = True  # 图片分布比例，-1图片点有重合，-2图片分布紧密但不行，1图片分布紧密但还可以，2图片分布比例完全合适
    if min_dis == 0 or max_dis == 0:
        scale = -1
    elif min_dis / max_dis < 1 / 100:
        for dis in dis_list.values():
            if math.sqrt(dis) * bl < 18:  # 最小距离小于20则比例不合适
                scale = -2
                break
        if scale != -2:
            scale = 1
    else:
        scale = 2

    # 6.3 画已知条件和结论中的元素
    edge_dict = {}  # 从语义信息中解析出所有的边，用来判断点的度是不是1
    # 已知条件
    # hypothesis='perpendicular(AC,BF);(CB,BA,FD,FG);'
    hypothesis = draw_geometric_element(img, edge_dict, point_draw, black, hypothesis, thickness, cv.LINE_AA)
    # 画结论中的元素
    if conclusion != '':
        conclusion = conclusion.split(';')[0]
        # conclusion='4_points_on_circle(A,C,D,G)'
        conclusion = draw_geometric_element(img, edge_dict, point_draw, red, conclusion, thickness, cv.LINE_AA)

    # 6.4 判断是否有孤立的点
    isolated_list = []
    for k, v in edge_dict.items():
        pt_list = list(v.keys())

        if len(pt_list) < 2:
            isolated_list.append(k)
        else:
            p1 = point_draw_old[k]
            isolated = True
            for com in combinations(pt_list, 2):
                try:
                    p2 = point_draw_old[com[0]]
                    p3 = point_draw_old[com[1]]
                    # |(y3−y1)(x2−x1)−(y2−y1)(x3−x1)|<=1e−6 两条直线的夹角
                    if Abs((p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) > 1e-6:
                        isolated = False
                        break
                except:
                    raise RuntimeError('孤立点检测出错')

            if isolated == True:
                isolated_list.append(k)

    isolated_point = None  # 是否有孤立点，0表示没有, 1表示有。有孤立点题目有平凡信息
    if len(isolated_list) > 0:
        for isolated in isolated_list:
            if re.search(r'IsBarycenterOf\({}'.format(isolated), hypothesis) != None:  # 重心
                isolated_point = 0
            elif re.search(r'IsCircumcenterOf\({}'.format(isolated), hypothesis) != None:  # 外心
                isolated_point = 0
            elif re.search(r'orthocenter\({}'.format(isolated), hypothesis) != None:  # 垂心
                isolated_point = 0
            else:
                isolated_point = 1
                break
    else:
        isolated_point = 0

    # 6.4  画点和点的标签
    point_size = 1
    point_in_hypothesis_conclusion = set(compile(r'[A-Z]').findall(hypothesis + conclusion))  # 只画题干和结论中的点
    for key, value in point_draw.items():
        if key not in point_in_hypothesis_conclusion:
            continue

        # dx = [0,   2, 5,  3,  0, -15, -15, -5]  # 字体大小为0.7 时 上 右上  右  右下 下 左下 左  左上
        # dy = [-3, -2, 0,  17, 20, 15, 0, -8]

        dx = [0, 3, 3, 2, 0, -12, -13, -5]  # 13  上 右上  右  右下 下 左下 左  左上
        dy = [-4, -2, 0, 10, 14, 12, 1, -5]

        # position_dic = {0: '上', 1: '右上', 2: '右', 3: '右下', 4: '下', 5: '左下', 6: '左', 7: '左上'}
        font_scale = 0.5  # 字体大小

        cv.circle(img, value, point_size, black, thickness)  # 先画点
        coox, cooy = value

        size = cv.getTextSize(key, fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=font_scale,
                              thickness=thickness)  # 字符写上去后占用的像素
        text_width = size[0][0]
        text_height = size[0][1]

        min_count = float('inf')  # 记录最少的非白色像素的个数
        drow_x = coox
        drow_y = cooy

        # mink = -1  # 记录标签最终在那个位置（上 右上  右  右下 下 左下 左  左上）
        for k in range(0, 8):
            d_x = coox + dx[k]
            d_y = cooy + dy[k]

            mask = np.array([255, 255, 255])  # 白色像素的值；基于 numpy 的矩阵特性，可以以矩阵为单位进行条件判断：
            recct = img[d_y - text_height:d_y,
                    d_x:d_x + text_width]  # 图形数据和numpy数组对应关系要求指定区域的时候，先是行，再是列；第2维是行，第1维是列；宽度对应列，高度对应行
            res = recct == mask  # 以矩阵为基础判断
            sss = np.all(res, axis=-1)  # 使用np.all()进行与操作，所有为true则为true。可把 4x4x3 维度的布尔条件映射到 4x4 的结果中。
            count = len(recct[sss])  # 白色像素的个数
            count = text_width * text_height - count  # 转换为不是白色的像素个数

            if count <= text_width * text_height / 10:  # 条件不设为0，设为1/10，让提前结束
                drow_x = d_x
                drow_y = d_y
                # print('提前结束')
                # if(count!=0):
                #     print('1111')
                break

            elif (min_count > count):
                # mink = k
                min_count = count
                drow_x = d_x
                drow_y = d_y

        # 打印一些结果，便于调试
        # print(key)
        # d_x = coox + dx[mink]
        # d_y = cooy + dy[mink]
        # img[d_y - text_height:d_y, d_x:d_x + text_width] = [0, 0, 0]  # 将某一块设置为黑色，演示numpy数组与图像像素数据的对应关系
        # cv.line(img, (d_x, d_y), (d_x + text_width, d_y - text_height), black, thickness) #画条直线，演示numpy数组与图像像素数据的关系
        # size = cv.getTextSize(key, fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=font_scale, thickness=thickness)
        # print(size)
        # print(key,'\t',position_dic[mink])
        # print(count)
        ##画点的标签
        cv.putText(img, key, (drow_x, drow_y), cv.FONT_HERSHEY_TRIPLEX, font_scale, blue, 1,
                   lineType)  # 字体就是这个，手再不要贱，不要修改。

    # 6.5 图形边缘留白裁剪
    mask = np.array([255, 255, 255])  # 白色像素的值；基于 numpy 的矩阵特性，可以以矩阵为单位进行条件判断：
    res = img != mask  # 以矩阵为基础判断
    sss = np.any(res, axis=-1)  # 对矩阵所有元素做或运算，存在True则返回True。可把 4x4x3 维度的布尔条件映射到 4x4 的结果中。 实现了降维
    coordinate_pair = np.argwhere(sss == True)  # 所有不是白色点的坐标，横纵坐标和图形数据是反的。
    coordinate_y = coordinate_pair[:, 0]
    coordinate_x = coordinate_pair[:, 1]

    min_x = np.min(coordinate_x)
    max_x = np.max(coordinate_x)
    min_y = np.min(coordinate_y)
    max_y = np.max(coordinate_y)

    # cv.circle(img, (minx, miny), point_size, black, 10)
    # cv.circle(img, (maxx, maxy), point_size, black, 10)

    # dy = min(int((max_y - min_y) / 8), 15)  # 留白最多是15个像素
    # dx = min(int((max_x - min_x) / 8), 15)
    dy = 15  # 留白最多是15个像素
    dx = 15
    # print(min_x, max_x, min_y, max_y)
    # print(dy,dx)
    # print('--------------------------------')

    boundary_min_y = max(0, min_y - dy)
    boundary_max_y = min(max_y, max_y + dy)
    boundary_min_x = max(0, min_x - dx)
    boundary_max_x = max(max_x, max_x + dx)

    img = img[boundary_min_y:boundary_max_y, boundary_min_x:boundary_max_x]
    # 等比例压缩，长、宽设置为500，400
    # img = img_resize(img)

    # 6.7 返回
    # scale -1图片点有重合，-2图片分布紧密但不行，1图片分布紧密但还可以，2图片分布比例完全合适。
    # isolated_point，0表示没有, 1表示有
    return (0, message, img, scale, isolated_point, point_draw, hypothesis, conclusion)


def get_characteristic(expression):
    '''
    根据一个表达式展开各项的特征，让这个表达式的正负确定，不再使用
    :param expression:
    :return:
    '''
    try:
        expression = sympify(expression, _clash, evaluate=False)
        expression = sympify(expand(expression))
        # 字母出现个数，由高到底排序
        letter_times_dic = dict(Counter(compile(r'([A-Z])').findall(str(expression))).most_common())
        subitem_times_dict = {}
        for i in range(len(expression.args)):
            times = 0
            for symbol in expression.args[i].free_symbols:
                times += letter_times_dic[str(symbol)]
            subitem_times_dict[i] = times

        return (letter_times_dic, subitem_times_dict)
    except Exception as e:
        1 == 1


#####  辗转相除法  求N个数的最大公约数和最小公倍数
def hcf(x, y):
    """该函数返回两个数的最大公约数"""
    if x >= y:
        max = x
        min = y
    if y > x:
        max = y
        min = x
    if (max % min) == 0:
        return min
    else:
        return (hcf(min, max % min))


def hcf1(x, length):
    """该函数返回多个数的最大公约数"""
    if (length == 1):
        return x[0]
    else:
        return hcf(x[length - 1], hcf1(x, length - 1))  # 递归实现


def lcm(x, y):
    ##该函数返回两个数的最小公倍数
    if (x % y == 0 or y % x == 0):
        if x >= y:
            return x
        if y > x:
            return y
    else:
        return (x * y) / hcf(x, y)


def lcm1(x, length):
    ##该函数返回多个数的最小公倍数
    if (length == 1):
        return x[0]
    else:
        return lcm(x[length - 1], lcm1(x, length - 1))


def get_hcf(num_list):
    """
    一组数中包含分数和整数，求这组数的最大公约数
    https://zhidao.baidu.com/question/151956962.html
    """
    # 获取绝对值最小值
    numerator = []  # 分子
    denominator = []  # 分母
    for x in num_list:
        x = Abs(x)
        mem, den = x.q, x.p
        numerator.append(int(den))
        denominator.append(int(mem))

    # 求分子的最大公约数
    num_hcf = hcf1(numerator, len(numerator))
    # print(num_hcf)

    # 求分母的最小倍数
    den_lcm = lcm1(denominator, len(denominator))
    # print(den_lcm)

    return sympify(num_hcf) / sympify(den_lcm)  # 分子的最大公约数 除以  分母的最小公倍数


def get_coefficient(expression):
    '''
    得到一个因式的系数
    :param expression:
    :return:
    '''
    coefficient = 1
    if (expression.is_Mul is True):
        for arg in expression.args:
            if arg.is_Number == True:
                coefficient = coefficient * arg
    elif expression.is_Symbol == True or expression.is_Pow == True:
        pass
    else:
        exit('输入的表达式应该是因式,或者为单个字母')

    return coefficient


def unified_formal_language(sentence):
    '''
    格式化形式语言，让参数可以互换的形式语言的表现字符唯一
    :param sentence:
    :return:
    '''
    if '=' in sentence:
        par1, par2 = sorted(sentence.split('='))
        sentence = r'equal({},{})'.format(par1, par2)
    else:
        index = sentence.find('(')
        if (index == -1):
            exit('几何形式语言有错HHH{}'.format(sentence))
        key = sentence[0:index]
        if key == 'perpendicular':
            result = compile(r'perpendicular\((\S*),(\S*)\)').search(sentence)
            try:
                line1, line2 = sorted(result.groups())
            except Exception as e:
                1 == 1
            sentence = r'perpendicular({},{})'.format(line1, line2)

        elif key in ['triangle', 'notcollinear', 'collinear']:
            result = compile(r'{}\(([A-Z]),([A-Z]),([A-Z])\)'.format(key)).search(sentence)
            p1, p2, p3 = sorted(result.groups())
            sentence = r'{}({},{},{})'.format(key, p1, p2, p3)

        elif key == '4_points_on_circle':
            result = compile(r'4_points_on_circle\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').search(sentence)
            p1, p2, p3, p4 = sorted(result.groups())
            sentence = r'4_points_on_circle({},{},{},{})'.format(p1, p2, p3, p4)
        elif key in ['quadrilateral', 'square', 'parallelogram', 'trapezoid']:  # 四边形
            result = compile(r'{}\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)'.format(key)).search(sentence)
            p1, p2, p3, p4 = sorted(result.groups())
            sentence = r'{}({},{},{},{})'.format(key, p1, p2, p3, p4)
        elif key in ['line_intersection_at']:  # 两线交点
            result = compile(r'line_intersection_at\((\S*),(\S*),([A-Z])\)').search(sentence)
            line1, line2 = sorted([result.groups()[0], result.groups()[1]])
            sentence = r'line_intersection_at({},{},{})'.format(line1, line2, result.groups()[2])
    return sentence


def unified_formal_language_times(sentence):
    '''
    格式化形式语言，让参数可以互换的形式语言的表现字符唯一。线段无法处理的
    :param sentence:
    :return:
    '''
    if '=' in sentence:
        par1, par2 = sorted(sentence.split('='))
        sentence = r'equal({},{})'.format(par1, par2)
    else:
        index = sentence.find('(')
        if (index == -1):
            exit('几何形式语言有错####{}'.format(sentence))
        key = sentence[0:index]
        if key == 'perpendicular':
            result = compile(r'perpendicular\((\S*),(\S*)\)').search(sentence)
            try:
                line1, line2 = sorted(result.groups())
            except Exception as e:
                1 == 1
            sentence = r'perpendicular({},{})'.format(line1, line2)
        elif key in ['triangle', 'notcollinear', 'collinear']:
            result = compile(r'{}\((\S*),(\S*),(\S*)\)'.format(key)).search(sentence)
            p1, p2, p3 = sorted(result.groups())
            sentence = r'{}({},{},{})'.format(key, p1, p2, p3)

        elif key == '4_points_on_circle':
            result = compile(r'4_points_on_circle\((\S*),(\S*),(\S*),(\S*)\)').search(sentence)
            p1, p2, p3, p4 = sorted(result.groups())
            sentence = r'4_points_on_circle({},{},{},{})'.format(p1, p2, p3, p4)
        elif key in ['quadrilateral', 'square', 'parallelogram', 'trapezoid']:  # 四边形
            result = compile(r'{}\((\S*),(\S*),(\S*),(\S*)\)'.format(key)).search(sentence)
            p1, p2, p3, p4 = sorted(result.groups())
            sentence = r'{}({},{},{},{})'.format(key, p1, p2, p3, p4)
        elif key in ['line_intersection_at']:  # 两线交点
            result = compile(r'line_intersection_at\((\S*),(\S*),(\S*)\)').search(sentence)
            line1, line2 = sorted([result.groups()[0], result.groups()[1]])
            sentence = r'line_intersection_at({},{},{})'.format(line1, line2, result.groups()[2])
    return sentence


def is_same(s1, s2):
    '''
    检测两个几何语句的含义是否相同，目前仅实现简易版
    :param s1:
    :param s2:
    :return:
    '''

    # 1.统一线段和直线
    pattern = compile(r'([A-Z])([A-Z])')
    for line in list(set(pattern.findall(s1))):
        old = ''.join(line)
        new = ''.join(sorted(line))
        if old != new:
            s1 = s1.replace(old, new)

    for line in list(set(pattern.findall(s2))):
        old = ''.join(line)
        new = ''.join(sorted(line))
        if old != new:
            s2 = s2.replace(old, new)

    # 2.格式化各个形式语言
    l1, l2 = [], []
    for s in s1.split(';'):
        if s.startswith('triangle') and '__' not in s:  # 过滤单个三角形,这里有特殊处理
            pass
        for ss in s.split('__'):
            ss = ss.strip()
            if ss == '':
                pass
            else:
                ss = unified_formal_language(ss)
                l1.append(ss)

    for s in s2.split(';'):
        if s.startswith('triangle') and '__' not in s:  # 过滤单个三角形
            pass
        for ss in s.split('__'):
            ss = ss.strip()
            if ss == '':
                pass
            else:
                ss = unified_formal_language(ss)
                l2.append(ss)

    # 通过集合判断，两边是否相同，或者一边属于另一边
    set_1, set_2 = set(l1), set(l2)
    if set_1.issubset(set_2) or set_2.issubset(set_1):
        print('一方是另一方的子集_包括一方为空的情况')  # 包括两方完全相同
        return True

    return False


def is_exists(existed_exercise: dict, one, exercise_name: str):
    '''
        判断新习题是否存在,目前仅实现简易版
    :param existed_exercise:  已存在的习题列表
    :param one: 需要判断习题的题设
    name: 需要判断的习题的名称
    :return:
    '''

    # 特殊调试
    # if exercise_name == '3.1_例_1--2---3.3_习_2--1---4.4':
    #     1 == 1
    # elif exercise_name == '3.1_例_1--2---3.3_习_9变3--2---4.3':
    #     1 == 1

    # 大写字母出现次数
    letter_times = dict(Counter(compile(r'([A-Z])').findall(str(one))))

    # 1.统一线段和直线，将它们中的字母先替换成出现的次数
    pattern = compile(r'([A-Z])([A-Z])')
    for line in list(set(pattern.findall(one))):
        old = ''.join(line)
        start, end = line
        start, end = letter_times[start], letter_times[end]
        new = '_'.join([str(s) for s in sorted([start, end])])
        one = one.replace(old, new)

    # 2.把其他语句种的大写字母替换为它出现的次数,替换其他大写字母
    for letter, times in letter_times.items():
        one = one.replace(letter, str(times))

    # 3.在大写字母出现次数的条件下，统一语句
    # 统一其他形式语言
    ll = []
    for s in one.split(';'):
        if s.startswith('triangle') and '__' not in s:  # 单个三角形不要
            pass
        for ss in s.split('__'):
            ss = ss.strip()
            if ss == '':
                pass
            else:
                ss = unified_formal_language_times(ss)
                ll.append(ss)

    ll = sorted(list(set(ll)))
    one = ';'.join(ll)

    if one in existed_exercise.keys():
        existed_exercise[one].append(exercise_name)
        print('已存在相同的习题')
        return True
    else:
        existed_exercise[one] = [exercise_name]
        return False


def find_repeat_with_old():
    '''
    检测与原始习题重复的习题
    :return:
    '''
    old_exercise = {}

    formula_list = pd.read_excel(r'D:\PycharmProjects\Assemble\Assemble\CSV\ch_3_split.xlsx',
                                 sheet_name='Sheet1')  # 原始题目信息
    old_list = formula_list['Eid'].drop_duplicates()
    # old_list=['3.3_例_10']
    # old_list=old_list[0:3]

    for exercise in old_list:
        idx = np.where(formula_list['Eid'] == exercise)[0]
        H = ';'.join(formula_list.loc[idx]['H1'].values.tolist())
        is_exists(old_exercise, H, exercise)

    new_list = pd.read_excel(r'./new_exercise.xlsx', sheet_name='Sheet1')
    delete_index = []
    for index, exercise in new_list.iterrows():
        H = exercise['H']
        if exercise['Name'] == '3.2_例_9--2---3.3_例_3--1---4.1':
            1 == 1
        # H, mark = exercise[['H', 'mark']]
        # H = '' if pd.isna(H) else H
        exercise_name = new_list.loc[index, 'Name']
        # if mark != '输出成功':
        #     continue
        if is_exists(old_exercise, H, exercise_name) == True:
            # print(exercise_name)
            delete_index.append(index)
    print('{}个与原始习题重复'.format(len(delete_index)))
    if len(delete_index) > 0:
        new_list = new_list.drop(index=delete_index)  # 删除与原始习题重复的
        new_list.to_excel(r'./new_exercise.xlsx', index=False)


def find_same_formula(pd_expression: pd):
    '''
    寻找同构的子式.  通过统计展开式（以及它的每个子式）中字母的出现次数以及 ，缩减了同构式子获取的耗时
    :param pd_expression: 子式的列表
    :return:
    '''

    # 1.给每个子式增加特征字符串
    pd_expression[['characteristic', 'times', 'expression_expand', 'letters']] = None, None, None, None  # 特征字符串

    # 只看部分习题
    # idx = np.where(datas['Eid'].isin(('3.2_例_9','3.3_习_11')))
    # datas = datas.loc[idx]
    # # # datas = datas.loc[[50, 67]]

    pd_expression = pd_expression.reset_index(drop=True)  # 重复建索引
    # 获取每个表达式的特征字符串
    for index, formula in pd_expression.iterrows():
        expression = list(formula['Expression'])[0]
        expression = simplify(expand(sympify(expression, _clash)))

        # 找最小公因数,让式子最小的系数为1
        coefficient_list = []
        for arg in expression.args:
            coefficient_list.append(get_coefficient(arg))
        hcf = get_hcf(coefficient_list)
        expression = expression / hcf
        # print(expression)

        letter_times_dic = dict(Counter(compile(r'([A-Z])').findall(str(expression))).most_common())  # 字母出现个数，
        subitem_times_dict = {}  # 子项中字母出现次数
        for i in range(len(expression.args)):
            times = 0
            for symbol in expression.args[i].free_symbols:
                times += letter_times_dic[str(symbol)]
            subitem_times_dict[i] = times

        letter_times = [str(s) for s in sorted(list(letter_times_dic.values()), reverse=True)]
        letter_times_str = '_'.join(letter_times)

        subitem_times = [str(s) for s in sorted(list(subitem_times_dict.values()), reverse=True)]
        subitem_times_str = '_'.join(subitem_times)

        characteristic = '{}@{}'.format(letter_times_str, subitem_times_str)

        pd_expression.loc[index, 'characteristic'] = characteristic
        pd_expression.at[index, 'letters'] = tuple(letter_times_dic.keys())  # loc不能赋值可迭代对象，at可以，因为at操作的是单个单元格
        pd_expression.at[index, 'times'] = tuple(letter_times_dic.values())
        pd_expression.loc[index, 'expression_expand'] = expression

    # 2.构建子式含义结构体
    # with open(r'./structure.pkl', 'rb') as File: #从外面读取
    #     structure = pickle.load(File)

    structure = pd.DataFrame(columns=['characteristic', 'sequence_index', 'value',
                                      'formula_idx'])  # sequence_index 点统一转换后的在A~Z的索引;formula_idx  子式在formula_idx 中的索引

    points = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
              'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # try:
    if 1 == 1:
        # 基于特征字符串寻找相同的表达式
        for charc in pd_expression['characteristic'].drop_duplicates():
            rows = np.where(pd_expression['characteristic'] == charc)[0].tolist()
            for row in rows:
                letters, times = pd_expression.loc[row, ['letters', 'times']]  # letters点序列，times 节点出现次数

                # 特征字符串在结构体中没有，添加新记录
                if structure[structure['characteristic'] == charc].values.size == 0:
                    idx = tuple(range(len(times)))
                    replace = {}
                    for id in idx:
                        replace[letters[id]] = points[id]

                    value = pd_expression.loc[row, 'expression_expand'].subs(replace, simultaneous=True)
                    new = [charc, value, [(row, idx)]]
                    new = pd.DataFrame(new, index=['characteristic', 'value', 'formula_idx']).T
                    structure = pd.concat([structure, new], ignore_index=True)

                else:  # 特征字符串在结构体中已有，开始表达式匹配
                    times_index = {}  # key为节点出现次数; value为列表，列表的元素为字母在letters（letters和times的索引一样）中的索引;相同出现次数的节点在一起
                    for i in range(len(times)):
                        ts = times[i]  # 字母出现次数
                        if ts not in times_index.keys():
                            times_index[ts] = [i]
                        else:
                            times_index[ts].append(i)

                    # 构造所有的可选的sequence_index，当多个点存在相同的次数，需要进行全排列
                    lll = [[]]
                    for index_list in list(times_index.values()):
                        if len(index_list) == 1:
                            for i in range(len(lll)):
                                lll[i] = lll[i] + index_list
                        else:
                            per_list = list(permutations(index_list, len(index_list)))  # 所有的排列
                            lll = copy.deepcopy(lll * len(per_list))
                            for i in range(len(per_list)):
                                lll[i] = lll[i] + list(per_list[i])

                    # 开始循环 匹配值
                    sequence_index = None
                    for ll in lll:
                        idx = tuple(ll)
                        sequence_index = idx
                        replace = {}
                        for iii in range(len(idx)):  # 20230403 修改。
                            id = idx[iii]
                            replace[letters[id]] = points[iii]

                        value = simplify(
                            expand(pd_expression.loc[row, 'expression_expand']).subs(replace, simultaneous=True))

                        filtered = np.where((structure['characteristic'] == charc) & (structure['value'] == value))[
                            0]  # 先匹配正值
                        if len(filtered) == 0:
                            filtered = np.where(
                                (structure['characteristic'] == charc) & (structure['value'] == value * -1))[
                                0]  # 再匹配相反数

                        if len(filtered) != 0:  # 匹配上,添加记录，然后跳出去
                            break

                    if len(filtered) == 0:  # 没有匹配上，往structure中添加记录
                        replace = {}
                        for id in range(len(times)):
                            replace[letters[id]] = points[id]

                        value = pd_expression.loc[row, 'expression_expand'].subs(replace, simultaneous=True)
                        new = [charc, value, [(row, sequence_index)]]
                        new = pd.DataFrame(new, index=['characteristic', 'value', 'formula_idx']).T
                        structure = pd.concat([structure, new], ignore_index=True)
                    else:  # 匹配上往structure 中附加记录
                        # print(row,Eid, value, replace)
                        structure.loc[filtered[0], 'formula_idx'].append((row, sequence_index))
    # except:
    #     1 == 1

    # with open(r'./structure.pkl', 'wb') as f:
    #     pickle.dump(structure, f)

    # ************输出这种方法对习题的分类方法，神经网调试时会用到
    # label_list = []
    # label = 0
    # for index, struc in structure.iterrows():
    #     label_list.append((label, struc['value'], struc['formula_idx'][0][0]))
    #     label += 1
    #
    # label_list = pd.DataFrame(label_list, columns=['lable', 'value', 'formula_id'])
    # path = r'D:\PycharmProjects\Assemble\datasets\Exercise\raw\Exercise_graph_label.csv'
    # label_list.to_csv(path, encoding='gbk', index=False, header=True)
    # return

    # 3.开始习题组装
    # print(structure)
    new_exercise = []
    repeated_exercise = {}

    # 统计撇配上的对
    # all_pair = 0
    # for index, struc in structure.iterrows():
    #     same_value_formulas = struc['formula_idx']
    #     if len(same_value_formulas) < 2:
    #         continue
    #
    #     for com in combinations(same_value_formulas, 2):
    #         try:
    #             formula_ids = [f[0] for f in com]
    #         except Exception:
    #             1 == 1
    #         sequence_1, sequence_2 = [f[1] for f in com]
    #
    #         Eids = datas.loc[formula_ids, 'Eid'].drop_duplicates()
    #         if len(Eids) < 2:  # 至少要两个习题之间组装;  筛选没有贡献的习题
    #             # print(Eids.values)
    #             continue
    #         else:
    #             all_pair += 1
    # print(all_pair)
    # return

    for index, struc in structure.iterrows():
        same_value_formulas = struc['formula_idx']
        if len(same_value_formulas) < 2:
            continue

        for com in combinations(same_value_formulas, 2):
            try:
                formula_ids = [f[0] for f in com]
            except Exception:
                1 == 1
            sequence_1, sequence_2 = [f[1] for f in com]

            Eids = pd_expression.loc[formula_ids, 'Eid'].drop_duplicates()
            if len(Eids) < 2:  # 至少要两个习题之间组装;  筛选没有贡献的习题
                # print(Eids.values)
                continue
            # else:
            #     continue

            f1, f2 = formula_ids
            F1, F2 = pd_expression.loc[f1], pd_expression.loc[f2]
            E1, E2 = F1['Eid'], F2['Eid']

            # if {E1, E2} != {'3.3_例_5', '3.3_习_26'}:  # 看两个习题 这才是集合
            #     continue

            # if '3.3_习_7' not in (E1, E2):  # 只看某个习题
            #     continue

            EX1, NO1, EX2, NO2 = E1, F1['NO'], E2, F2['NO']

            # 这里不应该排序的，因为两个不是等价的
            # p1, p2 = sorted([(EX1, NO1), (EX2, NO2)], key=lambda item: item[0], reverse=True)
            # exercise_name_old = '{}--{}---{}--{}'.format(p1[0], p1[1], p2[0], p2[1])
            exercise_name_old = '{}--{}---{}--{}'.format(EX1, NO1, EX2, NO2)
            # if exercise_name_old != '3.2_例_9--2---3.3_例_3--1':
            #     continue

            print('{}--{}'.format(EX1, NO1), '\t', '{}--{}'.format(EX2, NO2))
            'Eid', 'Point', 'Expression', 'Predicate', 'Eliminate', 'Draw'
            F1_E1, F1_H1, F1_NP1, F1_D1 = F1[['Expression', 'Predicate', 'Eliminate', 'Draw']]  # 1的正含义
            F1_E2, F1_H2, F1_NP2, F1_D2 = F1[
                ['Expression_Inverse', 'Predicate_Inverse', 'Eliminate_Inverse', 'Draw_Inverse']]  # 1的逆含义
            F2_E1, F2_H1, F2_NP1, F2_D1 = F2[['Expression', 'Predicate', 'Eliminate', 'Draw']]  # 2的正含义
            F2_E2, F2_H2, F2_NP2, F2_D2 = F2[
                ['Expression_Inverse', 'Predicate_Inverse', 'Eliminate_Inverse', 'Draw_Inverse']]  # 2的逆含义

            # 确定字母对应序列
            latter1, latter2 = [], []
            for i in sequence_1:
                latter1.append(F1['letters'][i])

            for i in sequence_2:
                latter2.append(F2['letters'][i])

            latter1, latter2 = ''.join(latter1), ''.join(latter2)

            print(F1_H1, ' ------- ', F1_H2)
            print(F2_H1, ' ------- ', F2_H2)
            E1_count, E2_count = len(np.where(pd_expression['Eid'] == E1)[0]), len(
                np.where(pd_expression['Eid'] == E2)[0])
            # E1_count, E2_count = 3, 3

            # 情况1, 1的逆含义 + 2的正含义 组成习题(2的正含义向1的逆含义对齐，下同)。 每对组合这种情况都有
            if F1_H2 != '':
                one_E, one_H, one_NP, one_D, another_E, another_H, another_NP, another_D = unify_letters(
                    F1_E2, F1_H2, F1_NP2, F1_D2, latter1,
                    F2_E1, F2_H1, F2_NP1, F2_D1, latter2)
                if is_same(one_H, another_H) == False:
                    E = ';'.join((one_E, another_E))
                    H = ';'.join((one_H, another_H))
                    NP = ';'.join((one_NP, another_NP))
                    D = ';'.join((one_D, another_D))

                    exercise_name = '{}---4.1'.format(exercise_name_old)
                    if is_exists(repeated_exercise, H, exercise_name) == False:
                        new_exercise.append((EX1, NO1, EX2, NO2, '4.1', exercise_name, E, H, NP, D))
                        print(exercise_name, '\n', E, '\n', H, '\n', '\n', NP, '\n', D, '\n')

            # 情况2, 2所在习题的子式个数大于2， 1的逆含义 + 2的逆含义组合
            if E2_count > 2 and F1_H2 != '' and F2_H2 != '':
                one_E, one_H, one_NP, one_D, another_E, another_H, another_NP, another_D = unify_letters(
                    F1_E2, F1_H2, F1_NP2, F1_D2, latter1,
                    F2_E2, F2_H2, F2_NP2, F2_D2, latter2)
                if is_same(one_H, another_H) == False:
                    E = ';'.join((one_E, another_E))
                    H = ';'.join((one_H, another_H))
                    NP = ';'.join((one_NP, another_NP))
                    D = ';'.join((one_D, another_D))
                    exercise_name = '{}---4.2'.format(exercise_name_old)
                    if is_exists(repeated_exercise, H, exercise_name) == False:
                        new_exercise.append((EX1, NO1, EX2, NO2, '4.2', exercise_name, E, H, NP, D))
                        print(exercise_name, '\n', E, '\n', H, '\n', '\n', NP, '\n', D, '\n')

            # 第三种情况，1所在的习题子式个数超过2, 2的逆含义 + 1的正含义组成习题; 2的逆含义 + 1的逆含义不能组合,和4.2 重复了
            if E1_count > 2 and F2_H2 != '':
                one_E, one_H, one_NP, one_D, another_E, another_H, another_NP, another_D = unify_letters(
                    F2_E2, F2_H2, F2_NP2, F2_D2,
                    latter2,
                    F1_E1, F1_H1, F1_NP1, F1_D1,
                    latter1)
                if is_same(one_H, another_H) == False:
                    E = ';'.join((one_E, another_E))
                    H = ';'.join((one_H, another_H))
                    NP = ';'.join((one_NP, another_NP))
                    D = ';'.join((one_D, another_D))
                    exercise_name = '{}---4.3'.format(exercise_name_old)
                    if is_exists(repeated_exercise, H, exercise_name) == False:
                        new_exercise.append((EX1, NO1, EX2, NO2, '4.3', exercise_name, E, H, NP, D))
                        print(exercise_name, '\n', E, '\n', H, '\n', '\n', NP, '\n', D, '\n')

            # 第四种情况，两边所在的习题都超过2，用1的正含义+2的正含义组成习题 (2的正含义是主体)
            if E1_count > 2 and E2_count > 2:
                one_E, one_H, one_NP, one_D, another_E, another_H, another_NP, another_D = unify_letters(
                    F2_E1, F2_H1, F2_NP1, F2_D1, latter2,
                    F1_E1, F1_H1, F1_NP1, F1_D1, latter1)
                if is_same(one_H, another_H) == False:
                    E = ';'.join((one_E, another_E))
                    H = ';'.join((one_H, another_H))
                    NP = ';'.join((one_NP, another_NP))
                    D = ';'.join((one_D, another_D))

                    exercise_name = '{}---4.4'.format(exercise_name_old)
                    if is_exists(repeated_exercise, H, exercise_name) == False:
                        new_exercise.append((EX1, NO1, EX2, NO2, '4.4', exercise_name, E, H, NP, D))
                        # print(exercise_name)
                        print(exercise_name, '\n', E, '\n', H, '\n', '\n', NP, '\n', D, '\n')

            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    # 把重复的习题存下来
    # with open(r'./repeated_exercise.pkl', 'wb') as f:
    #     pickle.dump(repeated_exercise, f)

    new_exercise = pd.DataFrame(new_exercise,
                                columns=['EX1', 'NO1', 'EX2', 'NO2', 'Type', 'Name', 'E', 'H', 'NP', 'D'])

    # 语句中去重
    for index, exercise in new_exercise.iterrows():
        E, H, NP, D, Name = exercise[['E', 'H', 'NP', 'D', 'Name']]

        # if Name == '3.3_例_3--3---3.3_例_12--1---4.2':
        #     1 == 1
        E = ';'.join(list(set(E.split(';'))))
        H = eliminate_repetition(H)
        NP = ';'.join(list(set(NP.split(';'))))
        D = eliminate_repetition(D)

        new_exercise.loc[index, ['E', 'H', 'NP', 'D']] = E, H, NP, D

    return (0, new_exercise)


def eliminate_repetition(paragraph):
    '''
    格式化形式语言，为了生成最后的习题
    :param sentence:
    :return:
    '''
    # 1.格式化线段
    pattern = compile(r'([A-Z])([A-Z])')
    for line in list(set(pattern.findall(paragraph))):
        old = ''.join(line)
        new = ''.join(sorted(line))
        if old != new:
            paragraph = paragraph.replace(old, new)

    new_paragraph = []
    for s in paragraph.split(';'):
        sentence_group = []
        for sentence in s.split('__'):
            sentence = sentence.strip()
            if sentence == '':
                pass
            else:
                if '=' in sentence:
                    par1, par2 = sorted(sentence.split('='))
                    sentence = r'equal({},{})'.format(par1, par2)
                    sentence_group.append(sentence)

                else:
                    index = sentence.find('(')
                    if (index == -1):
                        exit('几何形式语言有错123{}'.format(sentence))
                    key = sentence[0:index]
                    if key == 'perpendicular':
                        result = compile(r'perpendicular\(([A-Z][A-Z]),([A-Z][A-Z])\)').search(sentence)
                        try:
                            line1, line2 = sorted(result.groups())
                        except Exception as e:
                            1 == 1
                        sentence = r'perpendicular({},{})'.format(line1, line2)
                    elif key in ['triangle', 'notcollinear', 'collinear']:
                        result = compile(r'{}\(([A-Z]),([A-Z]),([A-Z])\)'.format(key)).search(sentence)
                        p1, p2, p3 = sorted(result.groups())
                        sentence = r'{}({},{},{})'.format(key, p1, p2, p3)

                    elif key == '4_points_on_circle':
                        result = compile(r'4_points_on_circle\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').search(sentence)
                        p1, p2, p3, p4 = sorted(result.groups())
                        sentence = r'4_points_on_circle({},{},{},{})'.format(p1, p2, p3, p4)
                    sentence_group.append(sentence)
        sentence_group = '__'.join(sentence_group)
        new_paragraph.append(sentence_group)
    new_paragraph = ';'.join(list(set(new_paragraph)))

    return new_paragraph


def unify_letters(one_E, one_H, one_NP, one_D, one_letters, another_E, another_H, another_NP,
                  another_D, another_letters):
    '''
    两部分要合并成一个习题，another并到one里面；处理题干中存在、展开式中不存在的点，对这类点重新命名，保证两边不一趟。
    :param one_H:
    :param one_NP:
    :param one_D:
    :param another_H:
    :param another_NP:
    :param another_D:
    :return:
    '''

    one_E, one_H, one_NP, one_D = ';'.join(list(one_E)), ';'.join(list(one_H)), ';'.join(list(one_NP)), ';'.join(
        list(one_D))
    another_E, another_H, another_NP, another_D = ';'.join(list(another_E)), ';'.join(list(another_H)), ';'.join(
        list(another_NP)), ';'.join(list(another_D))

    # 1.将题干中比展开式中多的点，用其他的字母代替,以消除重复问题
    s1 = set(re.compile(r'([A-Z])').findall(one_H))  # 题干中的大写字母
    s2 = set(one_letters)  # 开始式后的大写字母；s2肯定是s1 的子集

    s3 = set(re.compile(r'([A-Z])').findall(another_H))
    s4 = set(another_letters)  # s4肯定是s3的子集

    points = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
              'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    if len(s1 - s2) > 0 or len(s3 - s4) > 0:  # 删除points中两边都有的点
        for letter in tuple(s1.union(s3)):  # 在point中删除两边题干中都有的点
            points.remove(letter)

    if len(s1 - s2) > 0:
        replace = {}
        for lettet in tuple(s1 - s2):
            replace[lettet] = points.pop(0)

        map = str.maketrans(''.join(replace.keys()), ''.join(replace.values()))
        one_E = one_E.translate(map)
        one_H = one_H.translate(map)
        one_NP = one_NP.translate(map)
        one_D = one_D.translate(map)

    if len(s3 - s4) > 0:
        replace = {}
        for lettet in tuple(s3 - s4):
            replace[lettet] = points.pop(0)

        map = str.maketrans(''.join(replace.keys()), ''.join(replace.values()))
        another_E = another_E.translate(map)
        another_H = another_H.translate(map)
        another_NP = another_NP.translate(map)
        another_D = another_D.translate(map)

    if one_NP != '' and another_NP != '':  # 两边的衍生点都不为空,寻找是否有需要替换的衍生点
        NP_1_dic = {}
        for NP in one_NP.split(';'):
            nmae, value = NP.split('=')
            value = sympify(value, _clash)
            if value not in NP_1_dic.keys():
                NP_1_dic[value] = nmae
            else:
                exit('衍生点中出现重复的式子！！！！！！')

        NP_2_dic = {}
        for NP in another_NP.split(';'):  # another_letters 向one_letters 替换
            nmae, value = NP.split('=')
            value = sympify(value, _clash)
            if value not in NP_2_dic.keys():
                NP_2_dic[value] = nmae
            else:
                exit('衍生点中出现重复的式子！！！！！！')

        for va in NP_1_dic.keys():
            if va in NP_2_dic.keys():
                one_letters = one_letters + NP_1_dic[va]  # NP_1_dic[va] 不能在one_letters 中
                another_letters = another_letters + NP_2_dic[va]

    # 进行最后的替换 another向one看起
    map_finall = str.maketrans(another_letters, one_letters)  # 最后整体替换
    another_E = another_E.translate(map_finall)
    another_H = another_H.translate(map_finall)
    another_NP = another_NP.translate(map_finall)
    another_D = another_D.translate(map_finall)

    return (one_E, one_H, one_NP, one_D, another_E, another_H, another_NP, another_D)


def formula_output():
    '''
    画图, 画单个子式
    :return:
    '''
    formula_list = pd.read_excel(r'D:\PycharmProjects\Assemble\Assemble\CSV\ch_3_split.xlsx',
                                 sheet_name='Sheet1')  # 原始题目信息
    formula_list = formula_list.iloc[[52]]

    # 2.重置输出路径的文件
    path = r'./test'
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            pass
            # shutil.rmtree(c_path)
        else:
            if i.endswith('.png'):
                os.remove(c_path)

    with WolframLanguageSession() as session:
        with open("./test/index.html", 'w', encoding="UTF-8") as f_thml:
            f_thml.write(r'''<!DOCTYPE html>
                                                      <head>
                                                           <meta charset="UTF-8">
                                                           <title>New Exercises</title>
                                                           <meta http-equiv="X-UA-Compatible" content="IE=edge"/>
                                                           <meta name="viewport" content="width=device-width, initial-scale=1,maximun-scale=1,user-scalable=no"/>
                                                           <meta http-equiv="Pragma" content="no-cache"/>
                                                           <meta http-equiv="Cache-Control" content="no-cache"/>
                                                           <meta http-equiv="Expires" content="0"/>
                                                           <meta name="renderer" content="webkit">
                                                           <meta http-equiv="X-UA-Compatible" content="IE=EmulateIE7; IE=EmulateIE9"/>
                                                           <link href="./css/bootstrap.min.css" rel="stylesheet">
                                                           <script src="./js/jquery.min.js"></script>
                                                           <script src="./js/html5shiv.min.js"></script>
                                                           <script src="./js/respond.min.js"></script>
                                                           <link href="./css/style.css" rel="stylesheet">
                                                           <script type="text/javascript" async
                                                                   src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
                                                       </head>''')
            for index, formula in formula_list.iterrows():
                # if formula['Eid'] != '3.3_习_1':
                #     continue
                E1, H1, C1, NP1, D1 = formula[['E1', 'H1', 'C1', 'NP1', 'D1']]
                E1 = '' if pd.isna(E1) else E1
                H1 = '' if pd.isna(H1) else H1
                C1 = '' if pd.isna(C1) else C1
                NP1 = '' if pd.isna(NP1) else NP1
                D1 = '' if pd.isna(D1) else D1

                name = '{}_{}'.format(formula['Eid'], formula['NO'])

                if H1 != '':
                    result = Mapping(E1, NP1, H1, C1, D1, session)
                    id, message = result[0:2]

                    if id >= 0:
                        id, message, img, scale, isolated_point, point_draw, hypothesis, conclusion, error = result
                        # print(message)
                        if error != 0:
                            print(name)
                            print('误差检测未通过')
                        hypothesis = genereate_title(hypothesis)
                        # conclusion = genereate_title(conclusion)
                        # print(NP1)
                        # print(D1)
                        cv.imencode('.png', img)[1].tofile('./test/{}.png'.format(name))  # 土坪名称包含汉语

                        f_thml.write(r'''<div class="container list-main">
                                                    <div class='problem'>
                                                        <div class='txt'><b>子式{0}：</b> {1}</br>{2}</div>
                                                        <div class='img text-center'>
                                                            <img src="{2}.png">
                                                        </div>                                                            
                                                    </div>
                                                </div>'''.format(
                            index, hypothesis[1], name, E1))
                    else:
                        print(name)
                        print('画图失败', result[1])
                print('-------------------------------------------')


def exercise_output(exercise_list: pd):
    '''
    输出组装后的习题
    :return:
    '''
    # 控制习题输出
    # exercise_list = exercise_list.loc[list(range(12,13))]
    # exercise_list.sort_values(by="H", inplace=True, ascending=False)  # 验证了排序算法是稳定的

    exercise_list['out'] = None  # 输出结果

    # 2.重置输出路径的文件
    path = r'./new_exercises'
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            pass
            # shutil.rmtree(c_path)
        else:
            if i.endswith('.png'):
                os.remove(c_path)

    exercise_dict = {}  # 去重复

    # 3.开始输出习题
    with WolframLanguageSession() as session:
        with open("./new_exercises/index.html", 'w', encoding="UTF-8") as f_thml:
            f_thml.write(r'''<!DOCTYPE html>
                           <head>
                                <meta charset="UTF-8">
                                <title>New Exercises</title>
                                <meta http-equiv="X-UA-Compatible" content="IE=edge"/>
                                <meta name="viewport" content="width=device-width, initial-scale=1,maximun-scale=1,user-scalable=no"/>
                                <meta http-equiv="Pragma" content="no-cache"/>
                                <meta http-equiv="Cache-Control" content="no-cache"/>
                                <meta http-equiv="Expires" content="0"/>
                                <meta name="renderer" content="webkit">
                                <meta http-equiv="X-UA-Compatible" content="IE=EmulateIE7; IE=EmulateIE9"/>
                                <link href="./css/bootstrap.min.css" rel="stylesheet">
                                <script src="./js/jquery.min.js"></script>
                                <script src="./js/html5shiv.min.js"></script>
                                <script src="./js/respond.min.js"></script>
                                <link href="./css/style.css" rel="stylesheet">
                                <script type="text/javascript" async
                                        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
                            </head>''')

            for index, exercise in exercise_list.iterrows():
                EX1, NO1, EX2, NO2, Type, E, H, NP, D = exercise[
                    ['EX1', 'NO1', 'EX2', 'NO2', 'Type', 'E', 'H', 'NP', 'D']]
                E = '' if pd.isna(E) else E  # 表达式
                H = '' if pd.isna(H) else H  # 谓词语句
                NP = '' if pd.isna(NP) else NP  # 衍生点替换关系
                D = '' if pd.isna(D) else D  # 作图语句

                exercise_name = exercise_list.loc[index, 'Name']

                # 控制输出习题
                # if '3.3_习_26' not in (EX1, EX2):
                #     continue
                # if exercise_name not in ['0--1---1--3---4.2']:
                #     continue
                # if Type != 4.4:
                #     continue

                print('\n')
                print(exercise_name)

                C_list = list(set(H.split(';')).intersection(set(D.split(';'))))  # 这样设置，衍生点几何关系肯定不会被作为结论
                C_list_New = []
                for i in range(len(C_list)):
                    index = C_list[i].find('(')
                    if (index == -1):
                        raise RuntimeError('选定结论时出错————{}'.format(C_list[i]))
                    key = C_list[i][0:index]
                    if key not in ['quadrilateral', 'square', 'parallelogram', 'trapezoid', 'triangle', 'parallel',
                                   'collinear']:
                        C_list_New.append(C_list[i])

                C_list = C_list_New
                random.shuffle(C_list)
                for C in C_list:
                    delete_D = D.replace(C, '')
                    delete_H = H.replace(C, '')

                    print('----------------------')

                    print('已知条件:\t', delete_H)  # 格式化以后的语句
                    print('待求结论:\t', C)
                    print('作图语句:\t', delete_D)

                    if 1 == 1:
                        # try:
                        result = Mapping(E, NP, delete_H, C, delete_D, session)
                    # except Exception as e:
                    #     exercise_list.at[index, 'out'] = '出现异常！{}'.format(e)
                    #     print('\t 出现异常', e)
                    #     continue

                    id, message = result[0:2]
                    print('作图结果:\t', message)

                    if id == -10:
                        print('几何关系不相容，直接跳出')
                        exercise_list.loc[index, ['H', 'C', 'D', 'out']] = [delete_H, C, delete_D,
                                                                            '无法合成新习题']
                        break  # 第一次解方程无解，无法合成新习题，直接退出

                    elif id >= 0:
                        _, _, img, scale, isolated_point, point_draw, hypothesis, conclusion = result
                        print(result[1])
                        cv.imencode('.png', img)[1].tofile('./new_exercises/{}.png'.format(exercise_name))

                        exercise_list.loc[index, ['H', 'C', 'D', 'out']] = [hypothesis, conclusion, delete_D,
                                                                            '成功！{}'.format(message)]

                        hypothesis = genereate_title(hypothesis)[1]
                        conclusion = genereate_title(conclusion)[1]

                        title = '{}.&nbsp;&nbsp;求证:&nbsp;&nbsp;{}.'.format(hypothesis, conclusion)
                        f_thml.write(r'''<div class="container list-main">
                                                        <div class='problem'>
                                                            <div class='txt'> <b>习题&nbsp;&nbsp{0}&nbsp;&nbsp:&nbsp;&nbsp</b></br>{1}</br></div>
                                                            <div class='img text-center'>
                                                                <img src="{0}.png">
                                                            </div>                                                            
                                                        </div>
                                                    </div>'''.format(
                            exercise_name, title))
                        break  # 跳出去

    return(0,exercise_list)


def inverse_semantics(pd_expression: pd):
    '''
    获取子式的逆语义
    :param pd_expression: 表达式列表列表，以谓词语句呈现
    :return:
    '''

    pd_expression[
        ['Predicate_Inverse', 'Expression_Inverse', 'Eliminate_Inverse', 'Draw_Inverse']] = None, None, None, None
    for self, expression in pd_expression.iterrows():
        Eid = expression['Eid']

        expression_inverse, predicate_inverse, eliminate_inverse, draw_inverse = set(), set(), set(), set()

        for index in np.where(pd_expression['Eid'] == Eid)[0]:
            if index == self:
                continue
            one = pd_expression.loc[index]
            expression_inverse = expression_inverse.union(one['Expression'])
            predicate_inverse = predicate_inverse.union(one['Predicate'])
            eliminate_inverse = eliminate_inverse.union(one['Eliminate'])
            draw_inverse = draw_inverse.union(one['Draw'])

        # 做一次过滤， 正、逆含义不能相互包含;最正确的稍显条件应该是，一方推导另一方的时候不能有冗余
        if expression['Predicate'].issubset(predicate_inverse) == True or predicate_inverse.issubset(
                expression['Predicate']) == True:
            pd_expression.loc[self, ['Predicate_Inverse', 'Expression_Inverse', 'Eliminate_Inverse',
                                     'Draw_Inverse']] = set(), set(), set(), set()
            continue

        pd_expression.loc[self, ['Predicate_Inverse', 'Expression_Inverse', 'Eliminate_Inverse',
                                 'Draw_Inverse']] = predicate_inverse, expression_inverse, eliminate_inverse, draw_inverse

    pd_expression.to_excel(r'./inverse_result.xlsx', index=False)
    return (0, pd_expression)


if __name__ == "__main__":
    # 习题组装流程
    formula_output()

    with open(os.path.join(os.path.dirname(__file__), r'pd_expression.pkl'), 'rb') as f:
        pd_expression = pickle.load(f)

        # 3.获取子式的逆语义
    inverse_result = inverse_semantics(pd_expression)
    if inverse_result[0] != 0:
        exit('获取逆语义出错！')

    pd_expression = inverse_result[1]

    find_same_result = find_same_formula(pd_expression)
    if find_same_result[0] == 0:
        new_exercise = find_same_result[1]

    exercise_output(new_exercise)

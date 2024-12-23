#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/20 12:47
# @Author  : xuchaoyong
# @Site    :
# @File    : judge_convex_hull.py
# @Software: PyCharm
'''
凸包计算思路
https://blog.csdn.net/xydoo/article/details/104690296
https://www.cnblogs.com/aiguona/p/7232243.html
'''
import sys
import math
import time
import random





# 筛选出基准点,返回列表中基准点的标号，基准点是p[k]
def get_leftbottompoint(p):
    k = 0
    for i in range(1, len(p)):
        if p[i][1] < p[k][1] or (p[i][1] == p[k][1] and p[i][0] < p[k][0]):
            k = i
    return k


# 叉乘计算方法
def multiply(p1, p2, p0):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])


# 获取极角，通过求反正切得出，考虑pi/2的情况
def get_arc(p1, p0):
    # 兼容sort_points_tan的考虑
    if (p1[0] - p0[0]) == 0:
        if ((p1[1] - p0[1])) == 0:
            return -1;
        else:
            return math.pi / 2
    tan = float((p1[1] - p0[1])) / float((p1[0] - p0[0]))
    arc = math.atan(tan)
    if arc >= 0:
        return arc
    else:
        return math.pi + arc


# 对极角进行排序,排序结果list不包含基准点
def sort_points_tan(p, pk):
    p2 = []
    for i in range(0, len(p)):
        p2.append({"index": i, "arc": get_arc(p[i], pk)})
    # print('排序前:',p2)
    p2.sort(key=lambda k: (k.get('arc')))
    # print('排序后:',p2)
    p_out = []
    for i in range(0, len(p2)):
        p_out.append(p[p2[i]["index"]])
    return p_out


def convex_hull(p):
    '''

    :param p:
    :return: 凸包上的点 ，尽可能少的一些点，使这些点连起来构成的多边形可以把所有的点包含在里面。
    '''
    try:
        # p_old=p.copy() #20220901
        p = list(set(p))
        # print('全部点:',p)
        k = get_leftbottompoint(p)
        pk = p[k]
        p.remove(p[k])
        # print('排序前去除基准点的所有点:',p,'基准点:',pk)

        p_sort = sort_points_tan(p, pk)  # 按与基准点连线和x轴正向的夹角排序后的点坐标
        # print('其余点与基准点夹角排序:',p_sort)

        p_result = [pk, p_sort[0]]

        for i in range(1, len(p_sort)):
            #####################################
            # 叉乘为正,向前递归删点;叉乘为负,序列追加新点

                while (multiply(p_result[-2], p_sort[i], p_result[-1]) >0):
                    # if multiply(p_result[-2], p_sort[i], p_result[-1])>0:
                    #     p_result.pop()
                    # else:
                    #     p_result.append(p_sort[i])
                    p_result.pop()
                p_result.append(p_sort[i])

    except Exception  as e:
        raise Exception('__凸包计算出错！_{}'.format(str(e)))
        # p_result = list(set(p_old))
        # if len(p)==0: #20220901
        #     p_result=p
        # else:
        #     p_result = list(set(p_old))

    return p_result


def judge_convex(p):
    t = convex_hull(p)
    if len(t) == len(p):
        return True
    return False

if __name__ == "__main__":
    test_data = [(-1.00000000000000, -3.73205080756888), (0, 0), (1.00000000000000, -0.267949192431123), (0, -4.00000000000000), (-1.00000000000000, 0.267949192431123)]

    # # test_data = [(100, 100), (100, 200), (200, 100), (300, 100), (166, 133)]
    # test_data=[(1.00000000000000, 0.707106781186547),
    #            (0, 0),
    #            (3.00000000000000, -2.22044604925031e-16),
    #
    #            (-1.00000000000000, 1.41421356237310),
    #
    #            (0.333333333333333, 0.942809041582063)]
    # test_data = [(0, 0), (0.333333333333333, 0.942809041582063), (1.00000000000000, 0.707106781186547), (-1.00000000000000, 1.41421356237310)]
    # for i in range(len(test_data)):
    #     test_data[i]=(test_data[i][0],test_data[i][1])

    for i in range(1000):
        print(test_data)
        result = convex_hull(test_data)
        print(result)

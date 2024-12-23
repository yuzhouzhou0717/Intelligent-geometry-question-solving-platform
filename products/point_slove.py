from sympy import sympify, solve, symbols, simplify
from sympy.abc import _clash  # 类似 E、I　等冲突字母处理　
from re import compile, sub

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr
import random
from itertools import combinations
import copy
from collections import Counter

'''
利用mathematica 进行点几何恒等式求解 20230913
'''


def wolfram_evaluate(wolframc_str: str, session: WolframLanguageSession):
    '''
    调用mathematica内核,进行符号计算

    :param wolframc_str: mathematica 语句表示的方程和不等式组
    :param session: 调用mathematica 会话
    :return: （id,value）
        id<0  计算出现异常
        id==0 约束系统有解
        id==1 约束系统无解
    '''
    # print(wolframc_str)

    # 1.字符串中一些字符替换
    wolframc_str = wolframc_str.replace('**', '^')  # 幂运算替换

    # 2.设置运算超时限制
    wolframc_str = 'TimeConstrained[{},2]'.format(wolframc_str)  # 设置超时时间，默认时间3秒

    # 3.解约束系统
    try:
        result = session.evaluate(wlexpr(wolframc_str))
    except Exception as e:
        return (-1, '解约束系统出错___' + str(e))

    if hasattr(result, 'name') and result.name == '$Aborted':
        return (-2, '解约束系统超时')
    elif (result == '{}'):
        return (-3, '没有满足条件的解')
    else:
        return (0, result[2:-2])


def first_mine(title: list, hyp_dic: dict):
    '''
    输入题干，基于题干进行推理，分类得到几何关系，存储在hyp_dic中。 转换后得到的谓词语句要么可以转换为点几何表达式，要么为转换点几何恒等式服务

    推理使用使用的规则：
    1.特殊平行四边形（矩形、正方形等）转平行四边形、垂直、线段相等
    2.梯形转平行和线段相等
    3.中点、延长线转共线

    :param title: 题干
    :param hyp_dic: 分类存储几何关系,相等关系没有用谓词语句表示
    :return:
    '''
    hyp_dic['parallelogram'], hyp_dic['lieson'], hyp_dic['perpendicular'], hyp_dic['midpoint'] = [], [], [], []
    hyp_dic['equation'], hyp_dic['parallel'] = [], []
    hyp_dic['collinear'], hyp_dic['line_intersection_at'], hyp_dic['4_points_on_circle'] = [], [], []
    hyp_dic['circumcenter'] = []
    hyp_dic['both_side'], hyp_dic['triangle'], hyp_dic['square'] = [], [], []

    for hyp in title:
        if '=' in hyp:
            hyp_dic['equation'].append(hyp)
        else:
            index = hyp.find('(')
            if (index == -1):
                raise Exception('first_mine_谓词语句解析出错')
            key = hyp[0:index]

            if key in ('parallelogram', 'lieson', 'perpendicular', 'midpoint', 'line_intersection_at', 'parallel',
                       '4_points_on_circle', 'circumcenter', 'both_side', 'triangle', 'square'):  # 直接存储下来
                hyp_dic[key].append(hyp)
            if key == 'rectangle':  # rectangle(A,B,C,D)
                result = compile('rectangle\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').match(
                    hyp)
                p1, p2, p3, p4 = result.groups()
                hyp_dic['parallelogram'].append('parallelogram({},{},{},{})'.format(p1, p2, p3, p4))
                hyp_dic['perpendicular'].append('perpendicular({}{},{}{})'.format(p1, p2, p2, p3))
                hyp_dic['perpendicular'].append('perpendicular({}{},{}{})'.format(p2, p3, p3, p4))
                hyp_dic['perpendicular'].append('perpendicular({}{},{}{})'.format(p3, p4, p4, p1))
                hyp_dic['perpendicular'].append('perpendicular({}{},{}{})'.format(p4, p1, p1, p2))
            if key == 'square':  # square(A,B,C,D)
                result = compile('square\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').match(
                    hyp)
                p1, p2, p3, p4 = result.groups()
                hyp_dic['parallelogram'].append('parallelogram({},{},{},{})'.format(p1, p2, p3, p4))
                hyp_dic['perpendicular'].append('perpendicular({}{},{}{})'.format(p1, p2, p2, p3))
                hyp_dic['perpendicular'].append('perpendicular({}{},{}{})'.format(p2, p3, p3, p4))
                hyp_dic['perpendicular'].append('perpendicular({}{},{}{})'.format(p3, p4, p4, p1))
                hyp_dic['perpendicular'].append('perpendicular({}{},{}{})'.format(p4, p1, p1, p2))
                hyp_dic['equation'].append('{}{}={}{}'.format(p1, p2, p2, p3))
                hyp_dic['equation'].append('{}{}={}{}'.format(p2, p3, p3, p4))
                hyp_dic['equation'].append('{}{}={}{}'.format(p3, p4, p4, p1))
                hyp_dic['equation'].append('{}{}={}{}'.format(p4, p1, p1, p2))
            if key == 'equilateral_triangle':  # square(A,B,C,D)
                result = compile('equilateral_triangle\(([A-Z]),([A-Z]),([A-Z])\)').match(
                    hyp)
                p1, p2, p3 = result.groups()
                hyp_dic['equation'].append('{}{}={}{}'.format(p1, p2, p2, p3))
                hyp_dic['equation'].append('{}{}={}{}'.format(p1, p2, p1, p3))
                hyp_dic['equation'].append('{}{}={}{}'.format(p1, p3, p2, p3))

            # 把所有的共线的要找在一起;共线是集合的列表
            if key == 'collinear':
                points = set(compile('([A-Z])').findall(hyp))
                hyp_dic['collinear'].append(points)
            if key in ('lieson', 'midpoint'):
                result = compile('\(([A-Z]),([A-Z])([A-Z])\)').search(
                    hyp)
                points = set(result.groups())
                hyp_dic['collinear'].append(points)
            if key == 'line_intersection_at':
                result = compile('\(([A-Z])([A-Z]),([A-Z])([A-Z]),([A-Z])\)').search(
                    hyp)
                p1, p2, p3, p4, p5 = result.groups()
                hyp_dic['collinear'].append(set([p1, p2, p5]))
                hyp_dic['collinear'].append(set([p3, p4, p5]))

            if key == 'circumcenter':
                result = compile('circumcenter\(([A-Z]),triangle\(([A-Z]),([A-Z]),([A-Z])\)\)').search(
                    hyp)
                p1, p2, p3, p4 = result.groups()
                hyp_dic['equation'].append('{}{}={}{}'.format(p1, p2, p1, p3))
                hyp_dic['equation'].append('{}{}={}{}'.format(p1, p3, p1, p4))
                hyp_dic['equation'].append('{}{}={}{}'.format(p1, p2, p1, p4))

    # 点共线关系合并
    colliner_old, colliner_new = hyp_dic['collinear'], copy.deepcopy(hyp_dic['collinear'])
    while True:
        merge = False
        for com in combinations(colliner_old, 2):
            if len(com[0].intersection(com[1])) >= 2:
                colliner_new.remove(com[0])
                colliner_new.remove(com[1])
                colliner_new.append(com[0].union(com[1]))
                merge = True
        if merge == False:
            break
        else:
            colliner_old, colliner_new = colliner_new, copy.deepcopy(colliner_new)
    hyp_dic['collinear'] = colliner_new

    # 基于共线关系，形成两线相交关系
    intersection_add = []
    for com in combinations(hyp_dic['collinear'], 2):
        if len(com[0]) > 2 and len(com[1]) > 2:
            for point in set(com[0]).intersection(set(com[1])):
                for part1 in combinations(com[0].difference({point}), 2):
                    for part2 in combinations(com[1].difference({point}), 2):
                        if len(set(part1).intersection(set(part2))) == 0:
                            intersection_add.append(
                                'line_intersection_at({}{},{}{},{})'.format(part1[0], part1[1], part2[0], part2[1],
                                                                            point))
    hyp_dic['line_intersection_at'] = hyp_dic['line_intersection_at'] + intersection_add


def predicate_to_point_expression_first(hyp_dic: dict, sentence: str):
    '''
    谓词语句转换为点几何表达式，转换过程几何推理较少

    :param hyp_dic:
    :param sentence:
    :return:
    '''
    point_replace, expression = [], []  # 解题替换关系一次式子；其他几何关系是二次式子
    if '=' in sentence:  # 点几何表达式转等式
        sent_set = {sentence}  # 存储形成这个点几何表达式需要的谓词
        is_pass = False
        if compile(r'[\^\-\+]').search(sentence) == None:  # 只是两个线段之间的关系,需要转换为平方关系
            points = compile(r'([A-Z])').findall(sentence)
            set_points = set(points)
            line_pattern = compile(r'([A-Z])([A-Z])')

            for collinear in hyp_dic['collinear']:
                if set_points.issubset(collinear):  # 共线且相等的关系，应该转换为点的替换关系
                    is_pass = True
                    target = r'(\1-\2)'
                    sentence = sub(line_pattern, target, sentence)
                    sp = sentence.split('=')
                    eq = '{}-({})'.format(sp[0], sp[1])
                    eq = sympify(eq, _clash)

                    # point_times = Counter(points)

                    eq = sympify(eq, _clash)
                    sent_set = {sentence, 'collinear({})'.format(','.join(list(set_points)))}
                    ls = []
                    for p in list(set_points):
                        ls.append((p, str(solve(eq, symbols(p))[0])))

                    point_replace.append(
                        (ls, sent_set)
                    )

            if is_pass == False:  # 只是两条线段之间的关系，且这两条线段没有共线和平行，转成平方关系
                pattern = compile(r'([A-Z])([A-Z])')
                target = r'(\1-\2)'
                sentence = sub(pattern, target, sentence)
                sp = sentence.split('=')
                sentence = '({})^2=({})^2'.format(sp[0], sp[1])

                sp = sentence.split('=')
                expr = '{}-({})'.format(sp[0], sp[1])
                expression.append((expr, sent_set))  # 点几何表达式和谓词语句都存储
        else:
            for square in set(compile(r'([A-Z][A-Z]\^2)').findall(sentence)):  # 平方
                result = compile(r'([A-Z])').findall(square)
                text = '({}-{})^2'.format(result[0], result[1])
                sentence = sentence.replace(square, text)

            for mul in set(compile(r'([A-Z][A-Z]\*[A-Z][A-Z])').findall(sentence)):  # 乘积通过共线或者平方转换
                p1, p2, p3, p4 = compile(r'([A-Z])').findall(mul)
                points = set([p1, p2, p3, p4])
                for value2 in hyp_dic['collinear']:  # 检查是否共线
                    if points.issubset(value2):
                        text = '({}-{})*({}-{})'.format(p1, p2, p3, p4)  # 共线了就感这样写，但这里有问题，正负号的问题，只能在梳理种子题目时把线段方向统一
                        sentence = sentence.replace(mul, text)
                        break
                for value3 in hyp_dic['parallel']:  # 检查是否平行或者共线把
                    p11, p21, p31, p41 = compile('parallel\(([A-Z])([A-Z]),([A-Z])([A-Z])\)').match(
                        value3).groups()
                    if ({p1, p2} == {p11, p21} and {p3, p4} == {p31, p41}) or (
                            {p1, p2} == {p31, p41} and {p3, p4} == {p11, p21}):
                        text = '({}-{})*({}-{})'.format(p1, p2, p3, p4)  # 共线了就感这样写，但这里有问题，正负号的问题，只能在梳理种子题目时把线段方向统一
                        sentence = sentence.replace(mul, text)
                        break

            sp = sentence.split('=')
            expr = '{}-({})'.format(sp[0], sp[1])
            expression.append((expr, sent_set))  # 点几何表达式和谓词语句都存储
    else:
        index = sentence.find('(')
        if (index == -1):
            raise Exception('predicate_to_point_expression_first_谓词语句解析出错')
        key = sentence[0:index]

        if key == 'midpoint':  # midpoint(A,BC)
            result = compile('midpoint\(([A-Z]),([A-Z])([A-Z])\)').match(
                sentence)
            p1, p2, p3 = result.groups()
            point_replace.append(([(p1, '({}+{})/2'.format(p2, p3))], {sentence}))
        elif key == 'parallel':  # 平行，平行且相等
            result = compile('parallel\(([A-Z])([A-Z]),([A-Z])([A-Z])\)').match(
                sentence)
            p1, p2, p3, p4 = result.groups()
            for one in hyp_dic['equation']:
                if compile(r'[\^\-\+]').search(one) != None:  # 要求只是两个线段之间的一次关系
                    continue
                sp = one.split('=')
                part1 = compile('([A-Z])([A-Z])').search(sp[0]).groups()
                part2 = compile('([A-Z])([A-Z])').search(sp[1]).groups()

                # 等号两边的字母正好分别对应两条平行线
                match = False
                if {p1, p2} == set(part1) and {p3, p4} == set(part2):
                    match = True
                    part1_order, part2_order = {p1: 1, p2: 2}, {p3: 1, p4: 2}
                elif {p1, p2} == set(part2) and {p3, p4} == set(part1):
                    match = True
                    part2_order, part1_order = {p1: 1, p2: 2}, {p3: 1, p4: 2}
                if match == False:
                    continue

                # eq 标识要转换的等式，用来找点的替换关系
                if (part1_order[part1[0]] - part1_order[part1[1]]) * (
                        part2_order[part2[0]] - part2_order[part2[1]]) > 0:
                    eq = '{}-({})'.format(sp[0], sp[1])
                else:
                    eq = '{}+{}'.format(sp[0], sp[1])

                for line in compile(r'([A-Z][A-Z])').findall(eq):  # 乘积
                    text = '({}-{})'.format(line[0], line[1])
                    eq = eq.replace(line, text)

                sent_set = {sentence, one}
                point_replace.append(
                    (
                        [(p1, str(solve(eq, sympify(p1, _clash))[0])),
                         (p2, str(solve(eq, sympify(p2, _clash))[0])),
                         (p3, str(solve(eq, sympify(p3, _clash))[0])),
                         (p4, str(solve(eq, sympify(p4, _clash))[0]))],
                        sent_set
                    )
                )

        elif key == 'parallelogram':  # parallelogram(A,B,C,D)
            result = compile('parallelogram\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').match(
                sentence)
            p1, p2, p3, p4 = result.groups()
            point_replace.append(
                (
                    [(p1, '{}-{}+{}'.format(p2, p3, p4)),
                     (p2, '{}+{}-{}'.format(p1, p3, p4)),
                     (p3, '{}-{}+{}'.format(p2, p1, p4)),
                     (p4, '{}-{}+{}'.format(p1, p2, p3))
                     ],
                    {sentence}
                )
            )

        elif key == 'perpendicular':  # parallelogram(A,B,C,D)
            result = compile('perpendicular\(([A-Z])([A-Z]),([A-Z])([A-Z])\)').match(
                sentence)
            p1, p2, p3, p4 = result.groups()
            expr = '({}-{})*({}-{})'.format(p1, p2, p3, p4)
            sent_set = {sentence}
            expression.append((expr, sent_set))

        elif key == '4_points_on_circle':  # 圆幂定理 4_points_on_circle(A,E,C,F)
            result = compile('4_points_on_circle\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').match(
                sentence)
            p1, p2, p3, p4 = result.groups()
            for value2 in hyp_dic['line_intersection_at']:  # line_intersection_at(AE,CF,D)
                result2 = compile('line_intersection_at\(([A-Z])([A-Z]),([A-Z])([A-Z]),([A-Z])\)').match(
                    value2)
                p5, p6, p7, p8, p9 = result2.groups()
                if {p1, p2, p3, p4} == {p5, p6, p7, p8}:  # 这里有没有可能出错，所以应该以p5, p6, p7, p8为准
                    expr = '({}-{})*({}-{})-({}-{})*({}-{})'.format(p5, p9, p6, p9, p7, p9,
                                                                    p8, p9)
                    sent_set = {sentence, value2}
                    expression.append((expr, sent_set))

    return point_replace, expression


def predicate_to_point_expression_second(hyp_dic: dict):
    '''

    谓词语句转换为点几何表达式，转换过程有较多的推理
    :param hyp_dic: 已知题干中的几何关系，按类别存储
    :return:
    '''

    # 垂直关系通过共线传递,得到表达式
    point_replace, expression_add, expression_delete = [], [], {}
    for perpen in hyp_dic['perpendicular']:  # 垂直关系通过共线传递
        for collinear in hyp_dic['collinear']:
            if len(collinear) >= 3:  # AB垂直BC，且B、C、D共线，AB垂直CD ，还有（A-B）*(B-(C+D)/2)
                result = compile('perpendicular\(([A-Z])([A-Z]),([A-Z])([A-Z])\)').match(
                    perpen)
                p1, p2, p3, p4 = result.groups()
                if {p1, p2}.issubset(collinear):
                    for com in combinations(collinear, 2):
                        if {p1, p2} != com:
                            expr = '({}-{})*({}-{})'.format(com[0], com[1], p3, p4)
                            col_point = sorted(list(set((com[0], com[1], p1, p2))))
                            if len(col_point) > 2:
                                sent_set = {perpen, 'collinear({})'.format(','.join(col_point))}
                                expression_add.append((expr, sent_set))
                if {p3, p4}.issubset(collinear):
                    for com in combinations(collinear, 2):
                        if {p3, p4} != com:
                            expr = '({}-{})*({}-{})'.format(p1, p2, com[0], com[1])
                            col_point = sorted(list(set((com[0], com[1], p3, p4))))
                            if len(col_point) > 2:
                                sent_set = {perpen, 'collinear({})'.format(','.join(col_point))}
                                expression_add.append((expr, sent_set))

            if len(collinear) >= 3:  # AB垂直BC，且B、C、D共线，中垂线
                if {p1, p2}.issubset(collinear):
                    for com in combinations(collinear, 3):
                        if {p1, p2} != com:
                            expr = '({}-{})*({}-({}+{})/2)'.format(p3, p4, com[0], com[1], com[2])
                            col_point = sorted(list(set((com[0], com[1], p1, p2))))
                            sent_set = {perpen, 'collinear({})'.format(','.join(col_point))}
                            expression_add.append((expr, sent_set))
                if {p3, p4}.issubset(collinear):
                    for com in combinations(collinear, 3):
                        if {p3, p4} != com:
                            expr = '({}-{})*({}-({}+{})/2)'.format(p1, p2, com[0], com[1], com[2])
                            col_point = sorted(list(set((com[0], com[1], p3, p4))))
                            sent_set = {perpen, 'collinear({})'.format(','.join(col_point))}
                            expression_add.append((expr, sent_set))

    # 垂直关系通过平行传递
    for perpen in hyp_dic['perpendicular']:
        p1, p2, p3, p4 = compile('perpendicular\(([A-Z])([A-Z]),([A-Z])([A-Z])\)').match(
            perpen).groups()

        for parallel in hyp_dic['parallel']:
            p11, p21, p31, p41 = compile('parallel\(([A-Z])([A-Z]),([A-Z])([A-Z])\)').match(
                parallel).groups()
            if {p1, p2} == {p11, p21}:
                expr = '({}-{})*({}-{})'.format(p3, p4, p31, p41)
                sent_set = {perpen, parallel}
                expression_add.append((expr, sent_set))
            elif {p1, p2} == {p31, p41}:
                expr = '({}-{})*({}-{})'.format(p3, p4, p11, p21)
                sent_set = {perpen, parallel}
                expression_add.append((expr, sent_set))

            if {p3, p4} == {p11, p21}:
                expr = '({}-{})*({}-{})'.format(p1, p2, p31, p41)
                sent_set = {perpen, parallel}
                expression_add.append((expr, sent_set))

            elif {p3, p4} == {p31, p41}:
                expr = '({}-{})*({}-{})'.format(p1, p2, p11, p21)
                sent_set = {perpen, parallel}
                expression_add.append((expr, sent_set))

    # 等腰三角形，推导出几个垂直
    for equation in hyp_dic['equation']:
        result = compile('^([A-Z])([A-Z])=([A-Z])([A-Z])$').search(equation)  # 要求只是两个线段的关系。
        points = set(compile(r'([A-Z])').findall(equation))
        if result != None and len(points) == 3:  # 才有可能是等腰三角形
            point_times = Counter(result.groups())
            is_pass = False
            for lieson in hyp_dic['collinear']:  # 20231206 添加
                if points.issubset(lieson):  # 共线了，就不是等腰三角形了
                    is_pass = True
                    break

            if is_pass == False:
                vertex, base = None, []  # 等腰三角形的顶点和底边的两个顶点
                for point, times in point_times.items():
                    if times == 2:
                        vertex = point
                    else:
                        base.append(point)
                base1, base2 = base  # 底边的两个顶点
                # expr = '({}-({}+{})/2)*({}-{})'.format(vertex, base1, base2, base1, base2)  # 最基本的一个
                # sent_set = {equation}
                # expression_add.append((expr, sent_set))

                p1, p2, p3, p4 = result.groups()  # 要删除的几个.如果不删除，恒等式会冗长
                expression_delete['({}-{})^2-(({}-{})^2)'.format(p1, p2, p3, p4)] = 1

                for collinear in hyp_dic['collinear']:  # 垂直关系传递,产生中垂线
                    if {base1, base2}.issubset(collinear):
                        for com in combinations(collinear, 2):
                            if set(com) != {base1, base2}:
                                expr = '({}-({}+{})/2)*({}-{})'.format(vertex, base1, base2, com[0],
                                                                       com[1])  # 最基本的一个
                                col_point = sorted(list(set((com[0], com[1], base1, base2))))
                                sent_set = {equation, 'collinear({})'.format(','.join(col_point))}
                                expression_add.append((expr, sent_set))

    # 内积模型,向外做正方形、三角形等
    if len(hyp_dic['both_side']) > 1 and len(hyp_dic['square']) > 1 and len(hyp_dic['triangle']) > 0:
        for both_com in combinations(hyp_dic['both_side'], 2):
            for square_com in combinations(hyp_dic['square'], 2):
                for triangle in hyp_dic['triangle']:
                    tri_points = set(compile('triangle\(([A-Z]),([A-Z]),([A-Z])\)').match(triangle).groups())
                    squ_1_1, squ_1_2, squ_1_3, squ_1_4 = compile('square\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').match(
                        square_com[0]).groups()
                    squ_2_1, squ_2_2, squ_2_3, squ_2_4 = compile('square\(([A-Z]),([A-Z]),([A-Z]),([A-Z])\)').match(
                        square_com[1]).groups()
                    bot1_points = set(
                        compile('both_side\(([A-Z]),([A-Z]),([A-Z])([A-Z])\)').match(both_com[0]).groups())
                    bot2_points = set(compile('both_side\(([A-Z]),([A-Z]),([A-Z])([A-Z])\)').match(
                        both_com[1]).groups())
                    if bot1_points.union(bot2_points).issubset(
                            tri_points.union({squ_1_1, squ_1_2, squ_1_3, squ_1_4}).union(
                                {squ_2_1, squ_2_2, squ_2_3, squ_2_4})) == False:
                        continue

                    if squ_1_1 != squ_2_1:  # 必须要这样设置 ,两个矩形 4个定点是对应的
                        continue
                    com_point = squ_1_1
                    expr = '({}-{})*({}-{})-({}-{})*({}-{})'.format(squ_1_3, com_point, squ_2_2, com_point, squ_2_3,
                                                                    com_point, squ_1_2, com_point)  # 第一个式子
                    sent_set = {both_com[0], both_com[1], square_com[0], square_com[1], triangle}
                    expression_add.append((expr, sent_set))

                    expr = '({}-{})*({}-{})-({}-{})*({}-{})'.format(squ_1_4, com_point, squ_2_2, com_point, squ_2_4,
                                                                    com_point, squ_1_2, com_point)  # 第二个式子
                    sent_set = {both_com[0], both_com[1], square_com[0], square_com[1], triangle}
                    expression_add.append((expr, sent_set))

    return point_replace, expression_add, expression_delete  # 节点替换关系，增加的表达式


def slove_point(point_replace, hyp_expression, con_expression, session):
    '''
    求解点几何恒等式，求解过程中没有推理
    :param point_replace:  节点替换关系，一次式子
    :param hyp_expression: 已知条件中的二次式子
    :param con_expression: 待求结论中的二次式子
    :return: 系数解以及实际使用的点的替换关系
    '''
    equation_original, equation_finall = '', ''
    evaluate_result, point_replace_use = None, []  # 返回结果，系数解，实际使用的点的替换关系

    # 1.第一次尝试，不进行节点替换
    for i in range(len(hyp_expression)):
        equation_original = equation_original + '+k{}*({})'.format(i, hyp_expression[i][0])
    equation_original = equation_original + '+({})'.format(con_expression[0])

    variables = '{' + ','.join(set(compile(r'([A-Z])').findall(equation_original))) + '}'
    wolframc_str = 'Clear["Global`*"];ToString[SolveAlways[{0}==0,{1}],InputForm]'.format(
        equation_original.lower(),
        variables.lower())
    evaluate_result = wolfram_evaluate(wolframc_str, session)

    if evaluate_result[0] == 0:
        equation_finall = equation_original
    elif evaluate_result[0] == -1:  # 出错
        raise Exception('方程求解出错')
    elif evaluate_result[0] == -2:  # 超时
        pass
    elif evaluate_result[0] == -3:  # 没有满足要求的解,开始节点替换，进行第二次尝试
        for t in range(50):
            equation = equation_original
            point_replace_use = []  # 使用的节点替换关系，因为并不是所有替换关系都使用,是point_replace 的子集；最终输出是节点的替换顺序
            random.shuffle(point_replace)  # 节点替换关系就用随机的方法，因为 1：好多习题不受节点替换顺序没有影响，随机后以然成立；2：受节点替换顺序影响的习题，通过随机的方式成本最低。
            for i in range(len(point_replace)):
                if random.random() < 0.8:  # 衍生点替换关系是否采用，在这里控制；只要出现的节点替换关系一般会用，不用的情况较少，所以这个概率比较高是0.8;主要针对平行四边形
                    j = random.randint(0, len(point_replace[i][0]) - 1)  # 一条谓词语句对应的节点替换关系可以是多个，但只选一个
                    equation = equation.replace(point_replace[i][0][j][0],
                                                '(' + point_replace[i][0][j][1] + ')')
                    point_replace_use.append(
                        (point_replace[i][0][j][0], point_replace[i][0][j][1], point_replace[i][1]))

            variables = '{' + ','.join(set(compile(r'([A-Z])').findall(equation))) + '}'
            wolframc_str = 'Clear["Global`*"];ToString[SolveAlways[{0}==0,{1}],InputForm]'.format(
                equation.lower(), variables.lower())  # 把大写字母换成小写，避免冲突
            evaluate_result = wolfram_evaluate(wolframc_str, session)

            if evaluate_result[0] == 0:
                # print('{}次循环得到恒等式'.format(t + 1))
                equation_finall = equation
                break
            elif evaluate_result[0] == -1:  # 出错
                raise Exception('方程求解出错2')
            else:
                pass
    if equation_finall != '':
        return (0, evaluate_result, point_replace_use)
    else:
        return (-1, '求解失败')


def FindIdentity(hypothesis, conclusion, session):
    '''
    求解点几何恒等式，边推理边边计算。
    :param hypothesis: 已知道条件中的谓词语句
    :param conclusion: 待求结论中的谓词语句
    :param session:  mathematica会话
    :return:
    '''
    ' '
    if 1 == 1:
        # try:
        hypothesis = [hyp.replace(' ', '').replace(' ', '') for hyp in hypothesis]
        conclusion = [con.replace(' ', '').replace(' ', '') for con in conclusion]

        # 1.首先进行推理，获取几题干中的何关系
        hyp_dic = {}  # 题干中的几何关系，按类别存储，通过GPT和几何推理获取，为几何推理以及形成点几何表达式服务
        first_mine(hypothesis, hyp_dic)  # 挖掘的只是题干，结论不需要挖掘

        # 2.几何关系转点几何表达式
        point_replace, hyp_expression, con_expression = [], [], None  # 节点替换关系，一次式子；已知条件中的二次式子；待求结论中的二次式子
        for key, value_list in hyp_dic.items():
            if key in ('collinear'):
                continue
            for value in value_list:
                one_replace, one_expression = predicate_to_point_expression_first(hyp_dic, value)
                point_replace += one_replace
                hyp_expression += one_expression

        _, one_expression = predicate_to_point_expression_first(hyp_dic, conclusion[0])
        if len(one_expression) != 1:
            raise Exception('结论的式子个数有错')
        else:
            con_expression = one_expression[0]

        # 3.开始尝试求解恒等式
        slove_result = slove_point(point_replace, hyp_expression, con_expression, session)

        if slove_result[0] != 0:
            point_replace_add, hyp_expression_add, hyp_expression_delete = predicate_to_point_expression_second(
                hyp_dic)  # 开始第二次推理
            point_replace = point_replace + point_replace_add
            hyp_expression_new = []

            for hyp_exp in hyp_expression:
                if hyp_exp[0] not in hyp_expression_delete.keys():
                    hyp_expression_new.append(hyp_exp)
            hyp_expression = hyp_expression_new + hyp_expression_add

            slove_result = slove_point(point_replace, hyp_expression, con_expression, session)

        if slove_result[0] == 0:  # 恒等式求解成功,拼接恒等式
            evaluate_result, point_replace_use = slove_result[1:3]  # 形成恒等式时真正用的节点替换关系，是point_replace 的子集
            independent, dependent = [], {}  # 自变量系数，因变量系数
            coefficient_subs = {}  # 系数替换列表
            for spl1 in evaluate_result[1].split(','):
                spl2 = sympify(spl1.split('->'))
                if spl2[1].is_Number == True:
                    coefficient_subs[spl2[0]] = spl2[1]
                else:  # 让每个自变量为0，因变量先存列表，后面再赋值
                    dependent[spl2[0]] = spl2[1]
                    for symbol in spl2[1].free_symbols:
                        coefficient_subs[symbol] = 0
            # 因变量系数替换
            for k, v in dependent.items():
                coefficient_subs[k] = v.subs(coefficient_subs, simultaneous=True)

            # 用smpy包无法完成任务没办法输出好的结果，只能记录每个标号了，这个是必须要的，为了生成最终的答案，筛选多余的内容
            hyp_exprssion_index = {}  # 按序号标识，已知条件的表达式是否使用
            for i in range(len(hyp_expression)):
                hyp_exprssion_index[i] = 0
            for k, v in coefficient_subs.items():
                index = int(str(k)[1:])
                if v != 0:
                    hyp_exprssion_index[index] = v

            # 形成恒等式的每个子式，每个子式有谓词语句，为了习题嫁接
            subexpression_list = []  # 子式列表,每个子式的信息：点的名称，表达式，谓词语句，衍生点替换关系，作图语句
            subexpression_list.append(
                [
                    set(compile('([A-Z])').findall(con_expression[0])), str(sympify(con_expression[0], _clash)),
                    con_expression[1], set(), con_expression[1]
                ]
            )
            for k, v in hyp_exprssion_index.items():
                if v != 0:
                    subexpression_list.append(
                        [
                            set(compile('([A-Z])').findall(hyp_expression[k][0])),
                            str(sympify('{}*({})'.format(v, hyp_expression[k][0]), _clash)),
                            hyp_expression[k][1], set(), hyp_expression[k][1]
                        ]
                    )

            # 形成衍生点替换关系, 这里要嵌套处理，很麻烦，后面再处理
            point_replace_used_new = {}
            for repalce in point_replace_use:
                point_replace_used_new[repalce[0]] = (repalce[1], repalce[2])  # key是衍生点，值为：基本点表达式，谓词语句
            point_replace_use = point_replace_used_new

            for i in range(0, len(subexpression_list)):
                for point in point_replace_use.keys():
                    if point in subexpression_list[i][1]:  # 衍生点必须要这样统一替换,不然计算的时候恒等式等于0
                        subexpression_list[i][1] = subexpression_list[i][1].replace(point, '({})'.format(
                            point_replace_use[point][0]))
                        subexpression_list[i][2] = subexpression_list[i][2].union(point_replace_use[point][1])  # 合并谓词语句
                        subexpression_list[i][3] = subexpression_list[i][3].union(
                            {'{}={}'.format(point, point_replace_use[point][0])})  # 形成衍生点替换关系

            # 形成最终的恒等式,
            equation_finall = subexpression_list[0][1]
            for i in range(1, len(subexpression_list)):
                equation_finall = '{}+    {}'.format(equation_finall, subexpression_list[i][1])

            if simplify(sympify(equation_finall, _clash)) != 0:  # 验证恒等式是否为0
                return (-1, '恒等式不为0！')

            for i in range(0, len(subexpression_list)):  # 输出每个子式的信息
                subexpression_list[i][1] = {subexpression_list[i][1]}
                subexpression_list[i][3] = set(subexpression_list[i][3])
                # print(subexpression_list[i])

            return (0, equation_finall, subexpression_list, hyp_dic)

        else:
            return (-1, '恒等式求解失败！')

    # except Exception  as err:
    #     return (-1, '恒等式求解出错：' + str(err))


if __name__ == "__main__":
    hypothesis = ['triangle(A,B,C)', 'square(B,A,D,E)', 'square(B,C,G,F)', 'both_side(D,C,AB)', 'both_side(G,A,BC)',
                  'perpendicular(PG,AB)', 'perpendicular(PD,CB)']
    conclusion = ['perpendicular(PB,AC)']

    hypothesis = ["parallelogram(A,B,C,D)", "midpoint(M,BC)", "perpendicular(CP,AB)", "perpendicular(MA,MP)"]
    conclusion = ["PA=PD"]

    hypothesis = ["rectangle(A,B,C,D)", "equilateral_triangle(E,D,C)"]
    conclusion = ["AE=BE"]

    # hypothesis = ["triangle(A,B,C)", "AB=AC", "lieson(D,BC)", "lieson(E,AB)", "DB=ED"]
    # conclusion = ["4_points_on_circle(A,E,D,C)"]

    with WolframLanguageSession() as session:
        print(FindIdentity(hypothesis, conclusion, session)[1])
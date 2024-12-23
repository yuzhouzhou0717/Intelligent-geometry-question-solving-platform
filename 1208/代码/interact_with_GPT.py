import requests
import json


def analysis_exercise(title):
    '''
    利用chart-GPT4解析解析题意。
    原则，不进行向量推理，向量推理在点几何恒等式求解过程中实现
    体感中，共线点之间的关系需要人为指定
    :param title: 待解析习题题干
    :return:
    '''
    # 2023110 大模型无法使用,增加的功能
    an_result = {}
    an_result["如图，在矩形ABCD中，点E是AD的中点，过A、E、C三点的圆交直线CD于另一点F.求证:AF⊥BE."] = '''{
        "已知条件": ["rectangle(A,B,C,D)", "midpoint(E,AD)", "4_points_on_circle(A,E,C,F)","collinear(D,C,F)"],
        "待求结论": ["perpendicular(AF,BE)"]
    }'''

    an_result["如图，在△ABC中，AB=AC，D是BC上一点，E是AB上一点，AD⊥AC，DB=DE.求证：A、E、D、C四点共圆."] = '''{
      "已知条件": ["triangle(A,B,C)", "AB=AC", "lieson(D,BC)", "lieson(E,AB)", "perpendicular(AD,AC)", "DB=DE"],
      "待求结论": ["4_points_on_circle(A,E,D,C)"]
    }
    '''
    an_result["如图，在平行四边形ABCD中，M是BC中点，P是任意点（不一定在平面ABC上），如果CP⊥AB、MA⊥MP. 证明：PA=PD."] = '''{
      "已知条件": ["parallelogram(A,B,C,D)", "midpoint(M,BC)", "perpendicular(CP,AB)", "perpendicular(MA,MP)"],
      "待求结论": ["PA=PD"]
    } '''
    an_result["如图，在梯形ABCD中，AD∥BC，AD=3*BC，AB=AD，点M位于边CD上，且3*CM=2MD. 求证: BD⊥AM."] = '''{
      "已知条件": ["trapezoid(A,B,C,D)", "parallel(AD,BC)", "AD=3*BC", "AB=AD", "lieson(M,CD)", "3*CM=2*MD"],
      "待求结论": ["perpendicular(BD,AM)"]
    }    '''
    an_result["如图，平行四边形ABCD内，P是任意点，满足PC=BC，MN是连结线段AP,CD中点的直线.求证:MN⊥BP."] = '''{
          "已知条件": ["parallelogram(A,B,C,D)", "PC=BC","midpoint(M,AP)","midpoint(N,CD)"],
          "待求结论": ["perpendicular(MN,BP)"]
        } '''
    an_result["如图，梯形ABCD中，AD//BC，∠A=90°，E为AB中点，DE⊥CE.求证：AB^2=4*AD*BC."] = '''{
          "已知条件": ["trapezoid(A,B,C,D)", "parallel(AD,BC)", "perpendicular(AD,AB)", "midpoint(E,AB)", "perpendicular(DE,CE)"],
          "待求结论": ["AB^2=4*AD*BC"]
        }
        '''
    an_result["如图，D是AC延长线上一点，AB=AC=CD，E、F、G分别是AB、AC、BD中点，求证：FE⊥FG."] = '''{
          "已知条件": ["lieson(C,AD)","AB=AC", "AC=CD","midpoint(E,AB)", "midpoint(F,AC)", "midpoint(G,BD)"],
          "待求结论": ["perpendicular(FE,FG)"]
        } '''
    an_result["如图，AD⊥DB，AC⊥CB，E、F是AB、CD的中点.求证：EF⊥CD."] = '''{
          "已知条件": ["perpendicular(AD,DB)", "perpendicular(AC,CB)", "midpoint(E,AB)", "midpoint(F,CD)"],
          "待求结论": ["perpendicular(EF,CD)"]
        }'''
    an_result["如图，矩形ABCD，等边三角形EDC，连接AE、BE，求证：AE=BE."] = '''{
          "已知条件": ["rectangle(A,B,C,D)", "equilateral_triangle(E,D,C)"],
          "待求结论": ["AE=BE"]
        } '''
    # 10
    an_result["如图，三角形ABC中，∠BAC=90°，M、F、E分别为BC、CA、AB 的中点。求证:EF=AM."] = '''{
             "已知条件":["triangle(A,B,C)", "perpendicular(BA,AC)", "midpoint(M,BC)", "midpoint(F,CA)", "midpoint(E,AB)"],
             "待求结论": ["EF=AM"]
           } '''
    an_result["如图，由△ABC的两边向外做正方形BADE和BCGF，P是平面内一点，满足PG⊥AB、PD⊥CB. 求证：PB⊥AC."] = '''{
                     "已知条件": ["triangle(A,B,C)", "square(B,A,D,E)", "square(B,C,G,F)", "both_side(C,D,AB)", "both_side(G,A,BC)", "perpendicular(PG,AB)", "perpendicular(PD,CB)"],
                     "待求结论": ["perpendicular(PB,AC)"]
                   } '''
    an_result["如图，在等腰△ABC中，AB=AC，过C作CD⊥AB于D. 求证：BC^2=2*BD*BA."] = '''{
                             "已知条件": ["triangle(A,B,C)"t, "tAB=AC"t, "tlieson(D,AB)"t, "tperpendicular(CD,AB)"t],
                             "待求结论": ["tBC^2=2*BD*BA"t]
                           } '''
    an_result["如图，在△ABC中，AD、BE是高，F、G是AB、DE的中点.求证：FG⊥ED."] = '''{
                                 "已知条件": ["triangle(A,B,C)", "lieson(D,BC)", "perpendicular(AD,BC)", "lieson(E,AC)", "perpendicular(BE,AC)", "midpoint(F,AB)", "midpoint(G,DE)"],
                                 "待求结论": ["perpendicular(FG,ED)"]
                               } '''
    an_result["如图，△ABC中AB⊥BC，D是AC的中点，BC⊥ED. 求证: BE=EC."] = '''{
                                     "已知条件": ["triangle(A,B,C)", "perpendicular(AB,BC)", "midpoint(D,AC)", "perpendicular(BC,ED)"],
                                     "待求结论": ["BE=EC"]
                                   } '''
    # 15
    an_result["如图，在梯形ABCD中,AD//BC，∠ABC=90°,对角线BD⊥DC，求证:BD^2=AD*BC."] = '''{
                                         "已知条件": ["trapezoid(A,B,C,D)", "parallel(AD,BC)", "perpendicular(AB,BC)", "perpendicular(BD,DC)"],
                                         "待求结论": ["BD^2=AD*BC"]
                                       } '''
    an_result["如图，四边形ABCD中,M为BC中点,N为AC中点,P为DA中点,Q为DB中点.若AB=CD,试证:PM⊥QN."] = '''{
                                             "已知条件": ["quadrilateral(A,B,C,D)", "midpoint(M,BC)", "midpoint(N,AC)", "midpoint(P,DA)", "midpoint(Q,DB)", "AB=CD"],
                                             "待求结论": ["perpendicular(PM,QN)"]
                                           } '''
    an_result["如图，四边形ABCD中,M为BC中点,N为AC中点,P为DA中点,Q为DB中点.若AB=CD,试证:PM⊥QN."] = '''{
                                                 "已知条件": ["rectangle(A,B,C,D)", "parallel(PQ,AD)"],
                                                 "待求结论": ["AP^2+CQ^2=PB^2+DQ^2"]
                                               } '''
    an_result["如图，四边形ABCD中,M为BC中点,N为AC中点,P为DA中点,Q为DB中点.若AB=CD,试证:PM⊥QN."] = '''{
                                                     "已知条件":["4_points_on_circle(A,B,C,D)", "line_intersection_at(AC,BD,E)", "BE=ED"],
                                                     "待求结论":["AB^2+BC^2+CD^2+DA^2=2*AC^2"]
                                                   } '''
    an_result["如图，在△ABC中，AB=AC，D、E分别在边CA以及其延长线上，且CD=2AE，F是BD中点，求证：EF⊥BC."] = '''{
                                "已知条件":["triangle(A,B,C)", "AB=AC", "lieson(D,CA)", "lieson(A,ED)", "CD=2*AE", "midpoint(F,BD)"],
                                "待求结论":["perpendicular(EF,BC)"]
                                                       } '''
    #20
    an_result["如图，在△ABC中,AB=3BC,P、Q为边AB上两点,且满足PQ=AP=QB,点M是AC的中点，证明:∠PMQ=90°."] = '''{
                                    "已知条件":["triangle(A,B,C)", "AB=3*BC", "lieson(P,AB)", "lieson(Q,AB)", "PQ=AP", "AP=QB", "midpoint(M,AC)"],
                                    "待求结论":["perpendicular(PM,MQ)"]
                                                           } '''
    an_result["如图，在△ABC中,AB=AC,点D为BC的中点,连结AD,点E为AD的中点,作DG⊥BE于点G,点F为AC的中点. 求证：GF=DF."] = '''{
                                        "已知条件":["triangle(A,B,C)", "AB=AC", "midpoint(D,BC)", "midpoint(E,AD)", "perpendicular(DG,BE)", "collinear(G,B,E)", "midpoint(F,AC)"],
                                        "待求结论":["GF=DF"]
          } '''
    an_result["在△ABC中，AB⊥AC，在BC上取点P、Q，满足BA=BQ，CA=CP，求证：PQ^2=2*BP*QC."] = '''{
                                            "已知条件":["triangle(A,B,C)", "perpendicular(AB,AC)", "lieson(P,BC)", "lieson(Q,BC)", "BA=BQ", "CA=CP"],
                                            "待求结论":["PQ^2=2*BP*QC"]
              } '''
    an_result["如图，在梯形ABCD中,AD//BC,AB⊥BC，E为AB的中点,F为BC的中点,且AD=DC,求证:CE⊥DF."] = '''{
                "已知条件":["trapezoid(A,B,C,D)", "parallel(AD,BC)", "perpendicular(AB,BC)", "midpoint(E,AB)", "midpoint(F,BC)", "AD=DC"],
                "待求结论":["perpendicular(CE,DF)"]
                  } '''



    if title in an_result.keys():
        result = json.loads(an_result[title])
        if "已知条件" in result.keys() and "待求结论" in result.keys():
            return (0, result)
    else:
        #return (-1, "大模型请求超时123")
        pass

    system_prompt = r"""
        你是一名数学老师,对于给定的几何证明题,你的任务是解析习题的题意,即抽取已知体条件和待求结论中的几何关系,并将它们表示成谓词语句的形式。下表列出了常见几何关系的谓词表示形式。

        几何关系|谓词表示形式
        等腰三角形ABC中,AB＝AC | triangle(A,B,C),AB＝AC
        平行四边形ABCD | parallelogram(A,B,C,D)
        矩形ABCD | rectangle(A,B,C,D)
        菱形ABCD | diamond(A,B,C,D)
        正方形ABCD | square(A,B,C,D)
        四边形ABCD中，∠A=90° | trapezoid(A,B,C,D),perpendicular(BA,AD)
        A、B、C三点共线 | collinear(A,B,C)
        C在线段AB上 | lieson(C,AB)
        C是AB中点 | midpoint(C,AB)
        延长AB到C,使AB=2*BC | lieson(B,AC)；AB=2*BC
        延长BA到C,使AB=1/2*BC | lieson(A,BC)；AB=1/2*BC
        向两边延长AB到E、F,使得AE=AB=BF | lieson(A,EB),AE=AB,lieson(B,AF),AB=BF
        D是AC延长线上一点 | lieson(C,AD)
        线段AB等于CD | AB=CD
        AB=2CD | AB=2*CD 
        AB垂直CD | perpendicular(AB,CD)
        ∠EAF=90° | perpendicular(EA,AF)
        AB平行于CD | parallel(AB,CD)
        AB//CD | parallel(AB,CD)
        AB交CD于E | line_intersection_at(AB,CD,E)
        A、B、C、D四点共圆 | 4_points_on_circle(A,B,C,D)
        AD是三角形ABC的中线 | triangle(A,B,C),midpoint(D,BC)
        AD是三角形ABC的高 | lieson(D,BC),perpendicular(AD,BC)
        O是三角形ABC的外心 | circumcenter(O,triangle(A,B,C))
        △ABC的边BC延长至D | triangle(A,B,C),lieson(C,BD)
        等边三角形EFC | equilateral_triangle(E,F,C)
        C、D在线段AB的两侧 | both_side(C,D,AB) 
        过点O的直线EF分别交AD，BC于E、F两点 | collinear(E,O,F),collinear(A,E,D),collinear(B,C,F)
        DG⊥BE于点G | perpendicular(DG,BE),collinear(G,B,E) 
        在BC上取点P、Q  | lieson(P,BC),lieson(Q,BC)
        

        下面给出示例：
        示例1:
        输入的题目：在平行四边形ABCD中,AD=2AB,向两边延长AB到E、F,使得AE=AB=BF；求证：EC⊥FD.
        输出结果： {"已知条件":["parallelogram(A,B,C,D)","AB=2*AD","lieson(A,EB)", "AE=AB", "lieson(B,AF)", "AB=BF"],"待求结论":["perpendicular(EC, FD)"]}
        示例2：
        输入的题目：在矩形ABCD中,点E是AD的中点,过A、E、C三点的圆交直线CD于另一点F.求证：AF⊥BE。
        输出结果： {"已知条件":["rectangle(A,B,C,D)", "midpoint(E,AD)","4_points_on_circle(A,E,C,F)","collinear(D,C,F)"],"待求结论":["perpendicular(AF,BE)"]}
         
        注意事项：
        1.lieson(C,AB)表示点C在线段AB上，且点C在点A、B的中间。"延长BA到D"表示将线段AB沿B到A的方向延长至点D,那么A在B、D中间，所以应该为解析为lieson(A,BD).
        2.题意解析结果以JSON形式返回,JSON数据中仅包含已知条件和待求结论。已知条件和待求结论都是列表，且列表的每个元素仅包含1个谓词语句。
        3.所使用的谓词仅限表中列出的谓词语句，不要增加新的谓词。
        4."由△ABC的两边向外做正方形BADE、BCGF"，应该被解析为"triangle(A,B,C),square(B,A,D,E),square(B,C,G,F),both_side(C,D,AB),both_side(G,A,BC)",其中两个"both_side"用于确定"向外"信息。
        5."作DG⊥BE于点G"应该被解析为""，G在BE上的信息要通过"collinear(G,B,E)"说明。
        """

    user_first = ''' D、E是AB、AC上的点，延长CA到E，使AE=2CA.求证：AD=BE. '''
    assistant_first = ''' {"已知条件":["lieson(A,AE)","AE=2*CA"],"待求结论":["AD=BE"]} '''
    user_second = ''' 返回格式正确,但结果中 lieson(A,AE) 明显是错的，还有 "D、E是AB、AC上的点" 被遗漏了。 '''
    assistant_second = ''' {"已知条件":["lieson(D,AB)","lieson(E,AC)","lieson(A,CE)","AE=2*CA"],"待求结论":["AD=BE"]} '''

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_first})
    messages.append({"role": "assistant", "content": assistant_first})
    messages.append({"role": "user", "content": user_second})
    messages.append({"role": "assistant", "content": assistant_second})

    messages.append({"role": "user", "content": "请继续：\n" + title})


    # 连接大模型
    url = "https://research-cm.openai.azure.com/openai/deployments/GPT432K/chat/completions?api-version=2023-05-15"
    api_key = ""

    # 设置请求头
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    # 请求的数据
    data = {
        "messages": messages
    }

    # if 1 == 1:
    try:
        request_result = requests.post(url, headers=headers, json=data, timeout=10)  # 发送POST请求
        content = request_result.json()["choices"][0]["message"]["content"].strip()
        result = json.loads(content)
        if "已知条件" in result.keys() and "待求结论" in result.keys():
            return (0, result)
        else:
            return (-1, "GPT有相应,但是响应结果出错")
    except requests.exceptions.Timeout:
        return (-1, "大模型请求超时")
    except Exception  as err:
        return (-1, "大模型请求出错：" + str(err))


def polish_exercise(title):
    '''
    利用chart-GPT4润色习题题干
    :param title: 待解析习题题干
    :return:
    '''
    system_prompt = r"""
            一个AI程序可以生成几何证明题，但是它输出的题干不规范。你是一名数学教研员，你的任务是润色这个AI程序输出的题干，使其更规范、更简洁，达到教科书中所使用习题的水平。
            首先，下面列出了几道教科书中的几何证明题，你学习下它们的表述风格。
            1.如图，在矩形ABCD中,点E是AD的中点,过A,E,C三点的圆交直线CD于另一点F.求证:AF⊥BE. 
            2.如图，在△ABC中，AB=AC，延长AB到E，使得BE=AB，D是AB中点，求证：CE=2CD. 
            3.如图，圆O内接四边形ABCD，E、F、G分别是BC、AD、CD的中点，作平行四边形FOEN，求证：GN⊥AB. 
            4.如图，在△ABC中，CA⊥CB，AD、BE是中线，求证：AB^{2}+BE^{2}=\frac{5}{4} AB^{2}.
            5.如图，在梯形ABCD中,AD//BC. 在较长的底边BC上取一点E,使BE等于梯形的中位线长.求证:AC⊥BD的充要条件是DE也等于其中位线长.
            6.如图，在△ABC中,AB=AC,P为BC上一点.求证:  AB^{2}=A P^{2}+BP \cdot PC.

            然后，下表给出了一些润色题干的策略及示例，供你参考。                      
            策略 | 润色前 | 润色后
            将汉字描述的几何关系用几何符号表示。 | AB垂直于CD，GC平行于DF  | AB⊥CD，GC//DF
            依据几何知识，将一些几何关系融合，使题干更简洁。 | 平行四边形ABCD中，∠A=90° | 矩形ABCD
            使用并列句，使题干更简洁。 | A是BC的中点，D是EF的中点，平行四边形ABCD，圆内接四边形ABCD. | 圆内接平行四边形ABCD，A、D是BC、EF的中点.
            将以作图方式描述的几何关系展示在最前面 | 平行四边形ACBE，由△ADC的两边，向外作等边三角形AED、CDB. 求证：BE⊥EA. | 由△ADC的两边向外作等边三角形AED、CDB，如果四边形ACBE是平行四边形，求证：BE⊥EA.
            将数量关系转换为Latex文本，latex文本的的开头和结尾不需要"$" | AB^2=BC*AD,GB=2DC,ED=sqrt(2)*AE | AB^{2} = AD \cdot BC， GB = 2BC，ED=\sqrt{2}*AE
            数量关系中若出现分数，通过等式等价变形，使等式中不要出现分母，也不能出现小数 | AB=1/3*CD，XY=2/3*ZW，HG=0.5MN | 3AB=CD，ZW=3XY，MN=2HG
            将几何关系依据它们所涉及的几何对象的复杂程度由高到低呈现。几何对象的复杂程度排名（由高到低）： 圆，四边形(正方向，菱形，矩形，平行四变形，梯形，普通四边形)，三角形（等腰直角三角形，直角三角形，等边三角形，等腰三角形，普通三角形），线段共线(中点，三点共线，交点等)，平行，垂直，角度和线段之间的数量关系。 | A是CB的中点，CA//GF，平行四边形HFCB，FA交GC于D， | 平行四边形HFCB，FA交GC于D，A是CB的中点，CA//GF.            

            另外，下表给出了一些错误的示例，请你学习，不要犯类似的错误。
            待润色题干 | 错误结果 | 正确的结果 | 错因
            平行四边形CBDA，DE⊥EC. | 正方形CBDA | 矩形CBDA | 依据几何知识判断，CBDA只是矩形，不满足正方形的条件。这样的错误很严重，务必不要出现这样的错误！这种错误将导致习题出错，你要严格依据几何知识润色。
            A、C、D共线且CD=2AC，B是AD的中点. |直线AD上三点A、C、D共线,且CD=2AC,点B为直线AD上的中点. | 又是犯违背几何知识的错误，直线没有中点，线段才有！另外，题干要简洁，一些汉字完全可以删除。| 如图，B是AD中点，A、C、D共线CD=2AC.  
            平行四边形CBDA，BD⊥DA | 矩形ABCD | 矩形CBDA | 几何对象（尤其是多边形）名称中，字母的顺序不能修改。
            如图，C是DE的中点，F是BA的中点，BD⊥DA，BE⊥EA. 求证：FC⊥DE. | 如图，在BD⊥DA及BE⊥EA的条件下，C、F分别为DE、BA的中点，求证：FC⊥DE。 | 如图，BD⊥DA且BE⊥EA，C、F分别为DE、BA的中点，求证：FC⊥DE. | 题干应该简洁，不应出现"在...的条件下‘。还有，中文句号"。‘应替换成英文句号".‘. |
            矩形CBDA，DE⊥EC | 矩形CBDA中，DE⊥EC | 待润色的题干中，无法判断DE⊥EC在矩形中。如果待润色题干没有可润色的地方，则按原始内容输出，不能硬润色。 | 矩形CBDA，DE⊥EC


            """

    # user_first = ''' 平行四边形CBDA，DE⊥EC，BD⊥DA. 求证：BE⊥EA. '''
    # assistant_first = ''' 正方形CBDA中，线段DE垂直线段EC，求证：BE垂直AE。 '''
    # user_second = ''' 返回格式正确。但结果明显错误，CBDA是矩形，不是正方形，务必不要出现这样的错误，这些错误有毁灭性的后果，因为题干被这样修改后就成了一个错误的习题。
    #                   另外垂直、平行等术语要用几何符号表示。输出结果中不要出现"线段"，因为两个字母在一起就表示线段的含义。
    #                  '''
    # assistant_second = '''矩形ABCD，DE⊥EC，求证：BE⊥AE。 '''

    messages = [{"role": "system", "content": system_prompt}]
    # messages.append({"role": "user", "content": "\n" + user_first})
    # messages.append({"role": "assistant", "content": "\n" + assistant_first})
    # messages.append({"role": "user", "content": "\n" + user_second})
    # messages.append({"role": "assistant", "content": "\n" + assistant_second})
    messages.append({"role": "user", "content": "润色以下习题，结果存在JSON数组中，只返回润色后的题干，确保习题的返回顺序与习题的输入顺序一致。\n{}".format(title)})

    ####微软云大模型
    # 模型 GPT432K   终结点2023-05-15
    url = "https://research-cm.openai.azure.com/openai/deployments/GPT4/chat/completions?api-version=2023-05-15"
    # 设置请求头
    headers = {
        "Content-Type": "application/json",
        "api-key": ""
    }
    # 请求的数据
    data = {
        "messages": messages,
    }

    # if 1 == 1:
    try:
        request_result = requests.post(url, headers=headers, json=data)  # 发送POST请求
        content = request_result.json()["choices"][0]["message"]["content"].strip()
        return (content)
    except requests.exceptions.Timeout:
        return (-1, "大模型请求超时")
    except Exception  as err:
        # print(content)
        return (-1, "大模型请求出错：" + str(err))


if __name__ == "__main__":
    # exercise = "如图，D是AC延长线上一点，AB=AC=CD ,E、F、G分别是AB、AC、BD中点，求证：FE⊥FG."
    exercise = "平行四边形ABCD，连接AC，分别过B、D作AC的垂线交于E、F两点，求证：BE=DF。"
    # exercise = "三角形ABC中，∠BAC=90°,M、F、E分别为BC,CA,AB 的中点求证:EF =AM"
    exercise = "如图，矩形ABCD，等边三角形EDC，, 求证：AE=BE."
    result = analysis_exercise(exercise)
    if result[0] == 0:
        hypothesis, conclusion = result[1]["已知条件"], result[1]["待求结论"]
        print(result[1]["已知条件"])
        print(result[1]["待求结论"])
    else:
        print(result[1])

    from wolframclient.evaluation import WolframLanguageSession
    from point_slove import FindIdentity

    with WolframLanguageSession() as session:
        print(FindIdentity(hypothesis, conclusion, session)[1])

    exit()

    ######### 题干润色测试
    # import time
    #
    # old = []
    # old.append("如图，梯形CGAB中，GA//CB且GA=2CB，圆内接四边形CFEA，EA交CF于D，E是DA的中点，DA⊥AB，GD⊥DF. 求证：FA⊥EB.")
    # # old.append("平行四边形BCAE，圆内接四边形BCAE，CE交BA于D. 求证：BE⊥EA.")
    # # old.append("C是DE的中点，F是BA的中点，BD⊥DA，BE⊥EA. 求证：FC⊥DE.")
    # # old.append("如图，平行四边形AEBC，在△ADE的两边，向外作等边三角形ACD、EDB. 求证：BC⊥CA.")
    # old = "\n".join(old)
    #
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "\n")
    # new = polish_exercise(old)
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "\n")
    #
    # print(new)
    # with open("./result/polish.html", "w", encoding="UTF-8") as f_thml:
    #     f_thml.write(r'''<!DOCTYPE html>
    #                                        <head>
    #                                            <meta charset="UTF-8">
    #                                            <title>新习题</title>
    #                                            <meta http-equiv="X-UA-Compatible" content="IE=edge"/>
    #                                            <meta name="viewport" content="width=device-width, initial-scale=1,maximun-scale=1,user-scalable=no"/>
    #                                            <meta http-equiv="Pragma" content="no-cache"/>
    #                                            <meta http-equiv="Cache-Control" content="no-cache"/>
    #                                            <meta http-equiv="Expires" content="0"/>
    #                                            <meta name="renderer" content="webkit">
    #                                            <meta http-equiv="X-UA-Compatible" content="IE=EmulateIE7; IE=EmulateIE9"/>
    #                                            <link href="../css/bootstrap.min.css" rel="stylesheet">
    #                                            <script src="../js/jquery.min.js"></script>
    #                                            <script src="../js/html5shiv.min.js"></script>
    #                                            <script src="../js/respond.min.js"></script>
    #                                            <link href="../css/style.css" rel="stylesheet">
    #                                            <script type="text/javascript" async
    #                                                    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
    #                                        </head>''')  # 网页头写入
    #     f_thml.write(r'''<div class="container list-main">
    #                                                                               <div> {0}
    #                                                                                   </div>
    #                                                                                    <div> {1}
    #                                                                                   </div>
    #                                                                       </div>'''.format(old, "\( {} \)".format(new)))
